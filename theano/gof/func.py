
from .ifs import SimpleIFS
from .vm import Loop

class Function(object):
    def __init__(inputs, outputs,
                 updates={},
                 ifs_factory=SimpleIFS,
                 vm_factory=Loop):
        ivals = {v: None for v in inputs}
        internal_updates = dict(updates)
        internal_updates.update({v: v for v in outputs})
        self.inputs = inputs
        self.outputs = outputs
        self.ifs = ifs_factory(ivals, internal_updates)

    def __call__(self, *input_values):
        # -- TODO: check lists are same len
        for s_i, v_i in zip(self.inputs, input_values):
            self.ifs.set_value(s_i, v_i)
        self.ifs()
        return [self.ifs.get_value(v) for v in self.outputs]



