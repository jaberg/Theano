from .vm import Loop

def _vars_from_nodes(nodes)
    # -- N.B. variables must have deterministic order (no id/hash involved)
    variables = []
    seen = set()
    for node in nodes:
        for var in node.inputs + node.outputs:
            if var not in seen:
                variables.append(var)
                seen.add(var)
    return variables


class IterableFunctionSystem(object):
    """
    This class assumes that the graph connecting inputs, outputs and updates is constant.
    
    """
    def __init__(self, nodes, updates, int_from_ext, vm_factory=Loop):
        """
        ext_from_int - dictionary mapping from internally-used variables to externally defined
        ones
        """
        self._nodes = nodes
        self._updates = updates
        self._int_from_ext = int_from_ext
        self._vm_factory = vm_factory

        self._ext_from_int = {v: k for k, v in int_from_ext.items()}

        variables = _vars_from_nodes(self.nodes)
        self._storage_map = {v: [ivals.get(v)] for v in variables}
        self._compute_map = {v: (v.owner is None) for v in variables}

        if len(self._ext_from_int) != len(self._int_from_ext):
            raise ValueError('ext_from_int not 1-1', int_from_ext)
        for v, k in self._ext_from_int.items():
            if v.type != k.type:
                raise ValueError('ext_from_int type mismatch', (v, k))

    @property
    def nodes(self):
        """A list of the apply nodes of this IFS in a valid evaluation order
        """
        return self._nodes

    @property
    def variables(self):
        """A list of the variables involved in this IFS, for which values may be defined
        """
        return [self._ext_from_int.get(v, v) for v in _vars_from_nodes(self._nodes)]

    @property
    def updates(self):
        return self._updates

    def set_value(self, var, value, strict=False, **kwargs):
        """Set the value of an externally-defined variable
        """
        internal_var = self._int_from_ext.get(var, var)
        self.storage[internal_var][0] = var.type.filter(value, strict=strict, **kwargs)

    def get_value(self, var):
        """Set the value of an externally-defined variable
        """
        internal_var = self._int_from_ext.get(var, var)
        return self.storage[internal_var][0]

    def copy(self):

        # ifs's variables are the externally-defined ones
        # we will clone them, so that this instance has new internally-defined variables, and
        # the same set of external ones.

        int_from_ext = {}
        nodes = []
        for node in self.nodes:
            for internal_var in node.inputs:
                external_var = self._ext_from_int.get(internal_var, internal_var)
                if external_var not in int_from_ext:
                    int_from_ext[external_var] = external_var.clone()

            new_inputs = [int_from_ext[v] for v in node.inputs]
            new_node = node.clone_with_new_inputs(new_inputs)
            nodes.append(node)
            for old_internal_output, new_internal_output in zip(
                    node.outputs, new_node.outputs):
                external_output = self.ext_from_int.get(old_internal_output)
                int_from_ext[external_output] = new_internal_output

        return self.__class__(nodes, self.updates, int_from_ext)

    @property
    def vm(self):
        """Return a virtual machine to run this IFS
        """
        try:
            return self._vm
        except AttributeError:
            def make_thunk(node):
                return node.op.make_thunk(node, self._storage_map, self._compute_map,
                        no_recycling=[])
            thunks = map(make_thunk, self._nodes)
            self._vm = self.vm_factory(self._nodes, thunks, pre_call_clear=[])
            return self._vm

    def __call__(self, *args, **kwargs):
        # -- Compiles vm on demand
        return self.vm(*args, **kwargs)

    def __getstate__(self):
        rval = dict(self.__dict__)
        rval.pop('_vm')
        return rval


class SimpleIFS(IterableFunctionSystem):
    def __init__(self, ivals, updates, allow_auto_updates=True, vm_factory=vm.Loop):
        # XXX: rewrite so that updates is stable, not subject to random key order

        if allow_auto_updates:
            # -- repeatedly, recursively add updates until convergence
            raise NotImplementedError()
        else:
            _updates = dict(updates)

        inputs = list(ivals.keys())
        outputs = list(_updates.values())

        # -- correct when there are no destructive nodes
        nodes = graph.io_toposort(inputs, outputs)

        IterableFunctionSystem.__init__(self, vm, nodes, variables, updates, {})

    
