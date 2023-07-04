import types
from clrs._src import specs

Spec = specs.Spec

YZDSPECS = types.MappingProxyType({
    'yzd_liveness': {
        'pos': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'key': (specs.Stage.INPUT, specs.Location.NODE, specs.Type.SCALAR),
        'pred': (specs.Stage.OUTPUT, specs.Location.NODE, specs.Type.SHOULD_BE_PERMUTATION),
        'pred_h': (specs.Stage.HINT, specs.Location.NODE, specs.Type.POINTER),
        'i': (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE),
        'j': (specs.Stage.HINT, specs.Location.NODE, specs.Type.MASK_ONE)
    },
    'yzd_dominance': {},
    'yzd_reachability': {}
})
