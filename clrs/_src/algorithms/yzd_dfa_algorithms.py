import numpy as np

from clrs._src import yzd_utils
from clrs._src import yzd_specs, specs
from clrs._src import probing

def yzd_liveness(yzd_sample_loader: yzd_utils.SampleLoader,
                 sample_id: str):
    trace_list, array_list = yzd_sample_loader.load_a_sample(task_name='yzd_liveness',
                                                             sample_id=sample_id)
    cfg_array, gen_array, kill_array = array_list
    num_nodes = cfg_array.shape[0]
    probes = probing.initialize(spec=yzd_specs.YZDSPECS['yzd_liveness'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'pos': np.copy(np.arange(num_nodes)) * 1.0 / num_nodes,
                     'cfg': cfg_array,
                     'gen': gen_array,
                     'kill': kill_array,
                     'trace_i': trace_list[0]
                 })
    for time_idx in range(1, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': trace_list[time_idx],
                         'time': time_idx
                     }
                     )
    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': trace_list[-1]})
    probing.finalize(probes)
    return None, probes


def yzd_dominance(yzd_sample_loader: yzd_utils.SampleLoader,
                  sample_id: str):
    trace_list, array_list = yzd_sample_loader.load_a_sample(task_name='yzd_dominance',
                                                             sample_id=sample_id)
    cfg_array, gen_array = array_list
    num_nodes = cfg_array.shape[0]
    probes = probing.initialize(spec=yzd_specs.YZDSPECS['yzd_dominance'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'pos': np.copy(np.arange(num_nodes)) * 1.0 / num_nodes,
                     'cfg': cfg_array,
                     'gen': gen_array,
                     'trace_i': trace_list[0]
                 })
    for time_idx in range(1, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': trace_list[time_idx],
                         'time': time_idx
                     }
                     )
    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': trace_list[-1]})
    probing.finalize(probes)
    return None, probes


def yzd_reachability(yzd_sample_loader: yzd_utils.SampleLoader,
                     sample_id: str):
    trace_list, array_list = yzd_sample_loader.load_a_sample(task_name='yzd_reachability',
                                                             sample_id=sample_id)
    cfg_array, gen_array = array_list
    num_nodes = cfg_array.shape[0]
    probes = probing.initialize(spec=yzd_specs.YZDSPECS['yzd_reachability'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'pos': np.copy(np.arange(num_nodes)) * 1.0 / num_nodes,
                     'cfg': cfg_array,
                     'gen': gen_array,
                     'trace_i': trace_list[0]
                 })
    for time_idx in range(1, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': trace_list[time_idx],
                         'time': time_idx
                     }
                     )
    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': trace_list[-1]})
    probing.finalize(probes)
    return None, probes
