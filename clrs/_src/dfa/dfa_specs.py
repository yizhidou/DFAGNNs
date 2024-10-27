import types
from clrs._src import specs

# Spec = specs.Spec

DFASPECS = types.MappingProxyType({
    'others': { # DFA_GNN; DFA_GNN-
        'direction': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.CATEGORICAL),
        'cfg_edges': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.CATEGORICAL),
        'may_or_must': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.CATEGORICAL),
        'gen_vectors': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.CATEGORICAL),
        'kill_vectors': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.CATEGORICAL),
        'trace_h': (specs.Stage.HINT, specs.Location.NODE, specs.Type.CATEGORICAL),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.NODE, specs.Type.CATEGORICAL)},
    'plus': { # DFA_GNN+
        'direction': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.CATEGORICAL),
        'cfg_edges': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.CATEGORICAL),
        'may_or_must': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.CATEGORICAL),
        'gen_vectors': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.CATEGORICAL),
        'kill_vectors': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.CATEGORICAL),
        'trace_h': (specs.Stage.HINT, specs.Location.NODE, specs.Type.CATEGORICAL),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.NODE, specs.Type.CATEGORICAL)}
})
