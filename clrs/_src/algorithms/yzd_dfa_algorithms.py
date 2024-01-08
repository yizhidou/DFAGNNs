import numpy as np

from clrs._src import yzd_utils
from clrs._src import yzd_specs, specs
from clrs._src import probing, yzd_probing


def dfa(dfa_sample_loader: yzd_utils.SampleLoader,
        sample_id: str,
        task_name: str):
    '''sparse version'''
    trace_list, array_list = dfa_sample_loader.load_a_sample(task_name=task_name,
                                                             sample_id=sample_id)
    cfg_edges_forward, cfg_edges_backward, gen_vectors, kill_vectors = array_list
    if task_name == 'yzd_liveness':
        direction = np.ones()
        may_or_must = np.ones()
    elif task_name == 'yzd_dominance':
        direction = np.ones()
        may_or_must = np.ones()
    elif task_name == 'yzd_reachability':
        direction = np.ones()
        may_or_must = np.ones()
    else:
        raise yzd_utils.YZDExcpetion(yzd_utils.YZDExcpetion.UNRECOGNIZED_TASK_NAME)
    # num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=yzd_specs.DFASPECS['dfa'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'direction': direction,
                     'may_or_must': may_or_must,
                     'gen_vectors': gen_vectors,
                     'kill_vectors': kill_vectors,
                     'cfg_forward': cfg_edges_forward,
                     'cfg_backward': cfg_edges_backward
                 })
    # for time_idx in range(1, len(trace_list) - 1):
    for time_idx in range(0, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': trace_list[time_idx]
                     })

    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': trace_list[-1]})
    # expected_nb_nodes = dfa_sample_loader.max_num_pp + dfa_sample_loader.selected_num_ip
    expected_nb_cfg_edges = int(dfa_sample_loader.max_num_pp * dfa_sample_loader.cfg_edges_rate)
    edge_indices_dict, mask_dict = yzd_probing.dfa_finalize(probes=probes,
                                                            expected_nb_nodes=dfa_sample_loader.max_num_pp,
                                                            expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                            expected_hint_len=dfa_sample_loader.expected_hint_len)
    return edge_indices_dict, mask_dict, probes
