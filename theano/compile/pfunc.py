"""Provide a simple user friendly API """
__docformat__ = 'restructuredtext en'

import numpy # for backport to 2.4, to get any().

from profiling import ProfileStats
from theano.gof import Container, Variable, generic, graph, Constant, Value
from theano.compile import orig_function, In, Out
from theano.compile.sharedvalue import SharedVariable, shared
from theano import config
from theano import printing

import logging
_logger=logging.getLogger("theano.compile.pfunc")

class NoDefault:
    "Singleton object used by get() to indicate no default has been provided."

class MergeClone(object):
    """
    This class provides logic for merging together a graph specified
    via non-independent substitutions.

    The substitutions (aka givens) must be provided up-front to the constructor
    because this class must see all the substitutions before it can merge them
    together in the right order.

    clone_inputs - should generally be True, unless you know what you're doing
                It is possible to create cyclic replacement graphs if this is
                False

    """

    def __init__(self, inputs, outputs, replace_pairs, updates,
            rebuild_strict = True,
            clone_inputs = True,
            add_default_update = lambda var: True):
        for (old, new) in replace_pairs:
            if old in inputs:
                raise ValueError('replacing input is ambiguous')
        self._clone_d = {}         # old -> new mapping for merged output graph
        self.shared_inputs = []    # cloned shared variables in discovery order
        self.orig_inputs = inputs
        self.orig_outputs = outputs
        self.replace_pairs = replace_pairs
        self.updates = list(updates)
        self.updates_d = dict(updates)
        if len(self.updates) != len(self.updates_d):
            raise TypeError('duplicate shared var in updates list')
        self.clone_inputs = clone_inputs
        self.rebuild_strict = rebuild_strict
        self.add_default_update = add_default_update

    def has_cycles(self):
        newest_updates = [self.get_newest(repl)
                for (old, repl) in self.updates]
        print graph.io_toposort(self.built_inputs,
                self.built_outputs + newest_updates)

    def merge_all(self):
        self.merge_givens()
        print self._clone_d
        self.merge_inputs()
        self.merge_outputs()
        self.merge_update_expressions()
        print self.has_cycles()
        #if self.has_cycles():
            #raise ValueError('Merged graph contains cycles')

    def merge_givens(self):
        """
        Merge replace_pairs with the input->output graph (with updates)

        stores result as self.built_inputs, self.built_outputs, self._clone_d
        """
        self.sort_replace_pairs()
        for old, new in self.replace_pairs:
            newclone = self.clone_v_get_shared_updates(new)
            self.set_newest(old, newclone)

    def merge_inputs(self):
        for i in self.orig_inputs:
            if i in [old for (old, new) in self.replace_pairs]:
                raise ValueError('using givens to replace an input'
                        ' is ambiguous. Use the replacement in the'
                        ' inputs list if that is what you mean.',
                        shared_var)
            if not self.clone_inputs:
                self._clone_d.setdefault(i, i.clone())
            else:
                self._clone_d.setdefault(i, i)
        self.built_inputs = [self.get_newest(o) for o in self.orig_inputs]

    def merge_outputs(self):
        for output in self.orig_outputs:
            newclone = self.clone_v_get_shared_updates(output)
            self.set_newest(output, newclone)
        self.built_outputs = [self.get_newest(o) for o in self.orig_outputs]

    def merge_update_expressions(self):
        # Iterate over update_expr, cloning its elements, and updating
        # shared_inputs, update_d and update_expr from the SharedVariables
        # we discover.
        # If the variable to be updated is a shared variable not already
        # in shared_inputs, add it.
        i = 0
        # N.B. clone can append to update_expr, hence the use of `i`
        while i < len(self.updates):
            v, v_update = self.updates[i]
            self.clone_v_get_shared_updates(v)
            v_update = v.filter_update(
                    self.get_newest(v_update, v_update))
            if v_update.type != v.type:
                raise TypeError(
                    ('an update must have the same type as '
                      'the original shared variable'  ),
                    (v, v.type, v_update, v_update.type))
            newclone = self.clone_v_get_shared_updates(v_update)
            self.set_newest(v_update, newclone)
            i += 1

    def get_newest(self, key, fallback=NoDefault):
        try:
            rval = self._clone_d[key]
        except KeyError:
            if fallback is NoDefault:
                raise
            return fallback
        if rval != key:
            try:
                rval = self.get_newest(rval)
                self._clone_d[key] = rval
            except KeyError:
                pass
        return rval

    def set_newest(self, old, new):
        if old in self._clone_d:
            val = self._clone_d[old]
            if val is not old:
                self.set_newest(val, new)
            self._clone_d[old] = new
        else:
            self._clone_d[old] = new

    def has_update(self, key):
        return key in self.updates_d

    def get_update(self, key):
        return self.updates_d[key]

    def get_newest_update(self, key):
        return self.get_newest(self.updates_d[key])

    def clone_shared_input(self, v):
        if not isinstance(v, SharedVariable):
            raise TypeError(v)

        if v in self._clone_d:
            return self.get_newest(v)

        if self.clone_inputs:
            self._clone_d[v] = v.clone()
            cv = self._clone_d[v]
        else:
            cv = v
        self._clone_d[cv] = cv
        self.shared_inputs.append((v, cv))

        if v not in self.updates_d:
            if (self.add_default_update(v)
                    and hasattr(v, 'default_update')):
                orig_update_expr = v.default_update
                if orig_update_expr is not None:
                    self.updates_d[v] = orig_update_expr
                    self.updates.append((v, orig_update_expr))
                print 'adding update'
                printing.debugprint(orig_update_expr)

    def sort_replace_pairs(self):
        """
        Return a list of (oldvar, newvar) pairs in dependency order.

        returns: a list of [(old0, new0), (old1, new1), ...] pairs such that
        if A < B, then newA's does not depend on oldB.

        The purpose of this function is to support a sensible interpretation of
        givens when the various subgraphs they represent are tangled up and
        co-dependent.

        """
        # Suppose we're replacing vars v1 and v2,
        # but v2 appears in the ancestors of v1.
        # In this case we have to replace v2 first, and then v1.
        v_orig_ancestors = {}
        v_origs_set = set([v_orig for (v_orig, v_repl) in self.replace_pairs])
        for v_orig in v_origs_set:
            anc = graph.ancestors([v_orig],
                    blockers=set(self.orig_inputs
                        + [v for v in v_origs_set if v is not v_orig]))
            v_orig_ancestors[v_orig] = set(anc)
        def v_cmp(x, y):
            if x[0] in v_orig_ancestors[y[0]]:
                return -1
            if y[0] in v_orig_ancestors[x[0]]:
                return 1
            return 0
        self.replace_pairs.sort(v_cmp)

    def clone_v_get_shared_updates(self, v):
        '''
        Clones a variable and its inputs recursively until all are in
        _clone_d. Also appends all shared variables met along the way to
        shared inputs, and their default_update (if applicable) to update_d
        and update_expr.

        v can have an env attached to it, case in which we want to clone
        constants ( to avoid having a constant belonging to two envs)

        This function is co-recursive with self.clone_a(), for apply nodes
        '''
        assert v is not None
        if v not in self._clone_d:
            if v.owner:
                self.clone_a(v.owner) # inserts inputs and outputs into clone_d
            elif isinstance(v, SharedVariable):
                self.clone_shared_input(v)
            elif isinstance(v, Constant) and hasattr(v,'env'):
                # N.B. cloning constants does not copy the underlying object
                self._clone_d[v] = v.clone()
                v = self._clone_d[v]
                self._clone_d[v] = v
            elif self.clone_inputs:
                self._clone_d[v] = v.clone()
                v = self._clone_d[v]
                self._clone_d[v] = v
            else:
                self._clone_d[v] = v
        return self.get_newest(v)

    def clone_a(self, a):
        '''
        Clones a variable and its inputs recursively until all are in
        clone_d. It occures with clone_v_get_shared_updates
        '''
        if a is None:
            return None
        if a not in self._clone_d:
            for i in a.inputs:
                self.clone_v_get_shared_updates(i)
            self._clone_d[a] = a.clone_with_new_inputs(
                    [self.get_newest(i) for i in a.inputs],
                    strict = self.rebuild_strict)
            for old_o, new_o in zip(a.outputs, self._clone_d[a].outputs):
                self._clone_d.setdefault(old_o, new_o)
        return self.get_newest(a)


def rebuild_collect_shared(outputs,
        inputs             = None,
        replace            = None,
        updates            = None,
        rebuild_strict     = True,
        copy_inputs_over   = True,
        no_default_updates = False):
    """
    Function that allows replacing subgraphs of a computational
    graph.

    It returns a set of dictionaries and lists which collect (partial?)
    different information about shared variables. This info is required by
    `pfunc`.


    :type outputs: list of Theano Variables ( or Theano expressions)
    :param outputs: list of Theano variables or expressions representing the
                    outputs of the computational graph

    :type inputs: list of Theano Variables ( or Theano expressions)
    :param inputs: list of Theano variables or expressions representing the
                    inputs of the computational graph (or None)
    :type replace: dict
    :param replace: dictionary describing which subgraphs should be
                    replaced by what

    :type updates: dict
    :param updates: dictionary describing updates expressions for shared
                    variables

    :type rebuild_strict: bool
    :param rebuild_strict: flag, if true the type of all inputs should be
                            the same as the for the current node

    :type copy_inputs_over: bool
    :param copy_inputs_over: flag; if False it will clone inputs

    :type no_default_updates: either bool or list of Variables
    :param no_default_updates: if True, do not perform any automatic update
                               on Variables. If False (default), perform
                               them all. Else, perform automatic updates
                               on all Variables that are neither in
                               "updates" nor in "no_default_updates".

    """

    # inputs preparation
    if inputs is None:
        inputs = []

    # outputs preparation
    if isinstance(outputs,tuple):
        outputs = list(outputs)

    # replace preparation
    if replace is None:
        replace = []

    try:
        replace_pairs = replace.items()
    except Exception:
        replace_pairs = replace

    for i, (v_orig, v_repl) in enumerate(replace_pairs):
        if not isinstance(v_orig, Variable):
            raise TypeError('given keys must be Variable', v_orig)
        if not isinstance(v_repl, Variable):
            replace_pairs[i] = (v_orig, shared(v_repl))

    # It was decided to disallow shared variables from
    # being used as function inputs. Although it is technically possible,
    # it is also not clear when/how to use the value of that shared
    # variable (is it a default? ignored?, if the shared variable changes,
    # does that function default also change?).
    if numpy.any([isinstance(v, SharedVariable) for v in inputs]):
        raise TypeError(('Cannot use a shared variable (%s) as explicit '
                         'input. Consider substituting a non-shared'
                         ' variable via the `givens` parameter') % v)

    def var_from_output(v):
        if isinstance(v, Variable):
            return v
        elif isinstance(v, Out):
            return v.variable
        else:
            raise TypeError(
                    'outputs must be theano Variable or Out instances', v)

    if isinstance(outputs, list):
        output_vars = [var_from_output(o) for o in outputs]
    elif outputs == None:
        output_vars = []
    else:
        output_vars = [var_from_output(outputs)]

    # Fill update_d and update_expr with provided updates
    if updates is None:
        updates = []

    updates_vars = []
    for (shared_var, update_val) in iter_over_pairs(updates):
        if not isinstance(shared_var, SharedVariable):
            raise TypeError('update target must be a SharedVariable',
                    shared_var)
        if shared_var in [old for (old, new) in replace_pairs]:
            raise ValueError('Using a shared variable in both updates and givens'
                    ' is ambiguous. Use the replacement in the updates dict'
                    ' if that is what you mean.',
                    shared_var)
        if not isinstance(update_val, Variable):
            update_val = shared(update_val)

        if shared_var in [sv for (sv, val) in updates_vars]:
            raise ValueError('duplicate shared variable update', sv)
        updates_vars.append((shared_var, update_val))

    def add_default_update(v):
        try:
            return v not in no_default_updates
        except TypeError:
            return not no_default_updates
        
    # merge replace dict with inputs and outputs
    MC = MergeClone(inputs, output_vars, replace_pairs, updates_vars,
            clone_inputs=copy_inputs_over,
            add_default_update=add_default_update)

    MC.merge_all()

    # Elements of "outputs" are here cloned to "cloned_outputs"
    def out_from_cloned_v(cloned_v, v):
        if isinstance(v, Variable):
            return cloned_v
        elif isinstance(v, Out):
            return Out(cloned_v, borrow=v.borrow)
        elif v == None:
            return []
        else:
            assert 0, "This should have been caught above"

    if isinstance(outputs, list):
        cloned_outputs = [out_from_cloned_v(cloned_v, out)
                for cloned_v, out in zip(MC.built_outputs, outputs)]
    else:
        cloned_outputs = out_from_cloned_v(MC.built_outputs[0], outputs)

    return (MC.built_inputs, cloned_outputs, MC)


class Param(object):
    def __init__(self, variable, default=None, name=None, mutable=False,
            strict=False, allow_downcast=None, implicit=None, borrow = None):
        """
        :param variable: A variable in an expression graph to use as a compiled-function parameter

        :param default: The default value to use at call-time (can also be a Container where
        the function will find a value at call-time.)

        :param name: A string to identify this parameter from function kwargs.

        :param mutable: True -> function is allowed to modify this argument.

        :param borrow: True -> function is allowed to alias some output to
                       this input


        False: do not permit any output to be aliased to the input
        :param strict: False -> function arguments may be copied or casted to match the
        type required by the parameter `variable`.  True -> function arguments must exactly match the type
        required by `variable`.

        :param allow_downcast: Only applies if `strict` is False.
        True -> allow assigned value to lose precision when casted during assignment.
        False -> never allow precision loss.
        None -> only allow downcasting of a Python float to a scalar floatX.

        :param implicit: see help(theano.io.In)

        """
        self.variable = variable
        self.default = default
        self.name = name
        self.mutable = mutable
        # mutable implies the output can be both aliased to the input and that the input can be
        # destroyed. borrow simply implies the output can be aliased to the input. Thus
        # mutable=True should require borrow=True. Raise warning when borrow is explicitely set
        # to False with mutable=True.
        if mutable:
            if borrow==False:
                _logger.warning("Symbolic input for variable %s (name=%s) has "
                        "flags mutable=True, borrow=False. This combination is "
                        "incompatible since mutable=True implies that the "
                        "input variable may be both aliased (borrow=True) and "
                        "over-written. We set borrow=True and continue.",
                        variable, name)
            borrow = True
        self.strict = strict
        self.allow_downcast = allow_downcast
        self.implicit = implicit
        self.borrow = borrow


def pfunc(params, outputs=None, mode=None, updates=[], givens=[],
        no_default_updates=False, accept_inplace=False, name=None,
        rebuild_strict=True, allow_input_downcast=None,
        profile=None):
    """Function-constructor for graphs with shared variables.

    :type params: list of either Variable or Param instances.
    :param params: function parameters, these are not allowed to be shared
    variables

    :type outputs: list of Variables or Out instances
    :param outputs: expressions to compute

    :type mode: string or `theano.compile.Mode` instance.
    :param mode: compilation mode

    :type updates: iterable over pairs (shared_variable, new_expression). List, tuple or dict.
    :param updates: update the values for SharedVariable inputs according to these expressions

    :type givens: iterable over pairs (Var1, Var2) of Variables. List, tuple or dict.  The Var1
    and Var2 in each pair must have the same Type.

    :param givens: specific substitutions to make in the computation graph (Var2 replaces
    Var1).

    :type no_default_updates: either bool or list of Variables
    :param no_default_updates: if True, do not perform any automatic update on Variables.
    If False (default), perform them all. Else, perform automatic updates on all Variables
    that are neither in "updates" nor in "no_default_updates".

    :type name: None or string
    :param name: attaches a name to the Profiling result of this function when
    using ProfileMode (will be deprecated).

    :type allow_input_downcast: Boolean
    :param allow_input_downcast: True means that the values passed as
    inputs when calling the function can be silently downcasted to fit
    the dtype of the corresponding Variable, which may lose precision.
    False means that it will only be casted to a more general, or
    precise, type. None (default) is almost like False, but allows
    downcasting of Python float scalars to floatX.

    :type profile: None, True, str, or ProfileStats instance
    :param profile: accumulate profiling information into a given ProfileStats
    instance. None is the default, and means to use the value of
    config.profile.
    If argument is `True` then a new ProfileStats instance will be
    used.  If argument is a string, a new ProfileStats instance will be created
    with that string as its `message` attribute.  This profiling object will be
    available via self.profile.


    :rtype: theano.compile.Function
    :returns: a callable object that will compute the outputs (given the inputs)
    and update the implicit function arguments according to the `updates`.


    :note: Regarding givens: Be careful to make sure that these substitutions are
    independent--behaviour when Var1 of one pair appears in the graph leading to Var2 in
    another expression is undefined.  Replacements specified with givens are different from
    optimizations in that Var2 is not expected to be equivalent to Var1.

    """
    #
    # This function works by cloning the graph (except for the inputs), and then shipping it
    # off to compile.function
    # (There it will be cloned again, unnecessarily, because it doesn't know that we already
    # cloned it.)
    #
    # First, it clones the replacements named in the givens argument, and points each Var1 to
    # the clone of Var2.
    # Then it sets the inputs in the clone dictionary.
    # After these steps, we are assuming that the clone dictionary contains all the inputs to
    # the computation graph.
    #
    # Then it clones the outputs and the update expressions.  This rebuilds a computation graph
    # from the inputs and the givens.
    #
    if profile is None:
        profile = config.profile
        # profile -> True or False
    if profile == True:
        profile = ProfileStats(message=name)
        # profile -> object
    if type(profile) == str:
        profile = ProfileStats(message=profile)
    # profile is typically either False or an object at this point.
    # No need to block other objects being passed through though. It might be
    # useful.

    if not isinstance(params,(list,tuple)):
        raise Exception("in pfunc() the first argument must be a list or a tuple")

    if not isinstance(no_default_updates, bool)\
            and not isinstance(no_default_updates, list):
        raise TypeError("no_default_update should be either a boolean or a list")


    # transform params into theano.compile.In objects.
    inputs = [_pfunc_param_to_in(p, allow_downcast=allow_input_downcast)
              for p in params]

    in_variables = [ input.variable for input in inputs ]
    input_variables, cloned_outputs, MC = rebuild_collect_shared(outputs,
                            in_variables,
                            replace = givens,
                            updates = updates,
                            rebuild_strict = True,
                            copy_inputs_over = True,
                            no_default_updates = no_default_updates )

    for i, iv in zip(inputs, input_variables):
        i.variable = iv

    for sv, cv in MC.shared_inputs:
        if MC.has_update(sv):
            si = In(variable=cv,
                    value=sv.container,
                    mutable=True,
                    borrow=True,
                    update=MC.get_newest_update(sv))
        else:
            si = In(variable=cv,
                    value=sv.container,
                    mutable=False,
                    borrow=True)
        inputs.append(si)

    return orig_function(inputs, cloned_outputs, mode,
            accept_inplace=accept_inplace, name=name, profile=profile)


def _pfunc_param_to_in(param, strict=False, allow_downcast=None):
    if isinstance(param, Constant):
        raise TypeError('Constants not allowed in param list', param)
    #if isinstance(param, Value):
        #return In(variable=param)
        #raise NotImplementedError()
    if isinstance(param, Variable): #N.B. includes Value and SharedVariable
        return In(variable=param, strict=strict, allow_downcast=allow_downcast)
    elif isinstance(param, Param):
        return In(
                variable=param.variable,
                name=param.name,
                value=param.default,
                mutable=param.mutable,
                strict=param.strict,
                borrow = param.borrow,
                allow_downcast=param.allow_downcast,
                implicit = param.implicit)
    raise TypeError('Unknown parameter type: %s' % type(param))


def iter_over_pairs(pairs):
    """
    Return an iterator over pairs present in the 'pairs' input.

    :type pairs: dictionary or iterable
    :param pairs: The pairs to iterate upon. These may be stored either as
    (key, value) items in a dictionary, or directly as pairs in any kind of
    iterable structure

    :rtype: iterable
    :returns: an iterable yielding pairs

    """
    if isinstance(pairs, dict):
        return pairs.iteritems()
    else:
        return pairs
