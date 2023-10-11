import types
from clrs._src import specs

# Spec = specs.Spec

DFASPECS = types.MappingProxyType({
    'dfa_liveness': {
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
    'dfa_dominance': {
        'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'cfg': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'gen': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'kill': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'trace_i': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_h': (specs.Stage.HINT, specs.Location.EDGE, specs.Type.MASK),
        # 'time': (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.EDGE, specs.Type.MASK)},
    'dfa_reachability': {
        'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'cfg': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'gen': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'kill': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'trace_i': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_h': (specs.Stage.HINT, specs.Location.EDGE, specs.Type.MASK),
        # 'time': (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.EDGE, specs.Type.MASK)}
})
