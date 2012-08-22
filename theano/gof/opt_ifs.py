
from .ifs import IterableFunctionSystem

class OptimizingIFS(IterableFunctionSystem):

    def __init__(self, ifs, optimizer=None):


        # -- taking self.nodes to define the iterative computation
        self.fgraph = fg.FunctionGraph([], [])

        # -- TODO: steal this code from function_module
        #self.fgraph.extend( Supervisor(...))

        # If named nodes are replaced, keep the name
        self.fgraph.extend(toolbox.PreserveNames())

        for node in ifs.nodes:
            self.graph.

