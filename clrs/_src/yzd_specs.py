import types
from clrs._src import specs

Spec = specs.Spec

DFASPECS = types.MappingProxyType({
    'dfa': {
        'direction': (specs.Stage.INPUT, specs.Location.GRAPH, specs.Type.MASK),
        # 'may_or_must': (specs.Stage.INPUT, specs.Location.GRAPH, specs.Type.MASK),
        'gen_vectors': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.BINARY_MATRIX),
        'kill_vectors': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.BINARY_MATRIX),
        'cfg_edges': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        # 'cfg_backward': (specs.Stage.INPUT, specs.Location.EDGE, specs.Type.MASK),
        'trace_h': (specs.Stage.HINT, specs.Location.NODE, specs.Type.BINARY_MATRIX),
        'trace_o': (specs.Stage.OUTPUT, specs.Location.NODE, specs.Type.BINARY_MATRIX)}
})
