import types
from clrs._src import specs

Spec = specs.Spec

YZDSPECS = types.MappingProxyType({
    'yzd_liveness': {
        'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'cfg': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'gen': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'kill': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'if_pp': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.MASK),
        'if_ip': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.MASK),
        # 'adj': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.CATEGORICAL),
        'trace_i': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_h': (specs.Stage.HINT, specs.Location.EDGE, specs.Type.MASK),
        'time': (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.EDGE, specs.Type.MASK)
    },
    'yzd_dominance': {
        'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'cfg': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'gen': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'kill': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_i': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_h': (specs.Stage.HINT, specs.Location.EDGE, specs.Type.MASK),
        'time': (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.EDGE, specs.Type.MASK)},
    'yzd_reachability': {
        'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'cfg': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'gen': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'kill': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_i': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_h': (specs.Stage.HINT, specs.Location.EDGE, specs.Type.MASK),
        'time': (specs.Stage.HINT, specs.Location.GRAPH, specs.Type.SCALAR),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.EDGE, specs.Type.MASK)}
})
