import types

# specs.CLRS_30_ALGS_SETTINGS
# 其实我觉得这个在我的任务里好像是没必要的，因为看specs.py里对这个dict的定义，是只有一部分算法的值不为1的
# 所以为了弥补signal数量少的做法好像没啥必要就...
YZD_ALGS_SETTINGS = dict(yzd_liveness={'num_samples_multiplier': 1},
                         yzd_dominance={'num_samples_multiplier': 1},
                         yzd_reachability={'num_samples_multiplier': 1})

# samplers.CLRS30
YZDDFASamplerSettings = types.MappingProxyType({
    'train': {
        'num_samples': -1,
        'length': 16,
        'seed': 1,
        'max_step': 10,
        'sample_id_savepath': '',
    },
    'val': {
        'num_samples': -1,
        'length': 16,
        'seed': 2,
        'max_step': 10,
        'sample_id_savepath': '',
    },
    'test': {
        'num_samples': -1,
        'length': 64,
        'seed': 3,
        'max_step': 10,
        'sample_id_savepath': '',
    },
    'sourcegraph_dir': '',
    'dataset_savedir': ''
})
