import types
from clrs._src import specs

# Spec = specs.Spec

DFASPECS = types.MappingProxyType({
    'liveness': {
        'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'cfg': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'gen': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'kill': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'if_pp': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.MASK),
        'if_ip': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.MASK),
        # 'trace_i': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_h': (specs.Stage.HINT, specs.Location.EDGE, specs.Type.MASK),
        # 'time': (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.EDGE, specs.Type.MASK)
    },
    'dominance': {
        'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'cfg': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'gen': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'kill': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'trace_i': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_h': (specs.Stage.HINT, specs.Location.EDGE, specs.Type.MASK),
        # 'time': (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.EDGE, specs.Type.MASK)},
    'reachability': {
        'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'cfg': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'gen': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'kill': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'trace_i': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_h': (specs.Stage.HINT, specs.Location.EDGE, specs.Type.MASK),
        # 'time': (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.EDGE, specs.Type.MASK)},
    'dfa': {
        # 'may_or_must': (specs.Stage.INPUT, specs.Location.GRAPH, specs.Type.MASK),
        'direction': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'cfg_edges': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'gen_vectors': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.BINARY_MATRIX),
        'kill_vectors': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.BINARY_MATRIX),
        'trace_h': (specs.Stage.HINT, specs.Location.NODE, specs.Type.BINARY_MATRIX),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.NODE, specs.Type.BINARY_MATRIX)}
})
