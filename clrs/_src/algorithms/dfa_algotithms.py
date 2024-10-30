import numpy as np

from clrs._src import dfa_utils
from clrs._src import dfa_specs, specs
from clrs._src import probing, dfa_probing

def dfa_others(dfa_sample_loader: dfa_utils.SampleLoader,
           sample_id: str,
           task_name: str):
    '''sparse version'''
    trace_list, array_list, full_trace_len = dfa_sample_loader.load_a_sample(task_name=task_name,
                                                             sample_id=sample_id)
    gen_vectors, kill_vectors, cfg_edges = array_list
    # print(f'algo line 243, gen_vectors: {gen_vectors.shape}')
    # [num_pp, selected_num_ip ]
    num_ip = gen_vectors.shape[1]
    # may_or_must = np.zeros((num_ip, 2))
    if task_name == 'liveness':
        direction = 0
        may_or_must = 0
        # may_or_must[:, 0] = 1
    elif task_name == 'dominance':
        direction = 1
        may_or_must = 1
        # may_or_must[:, 1] = 1
    elif task_name == 'reachability':
        direction = 0
        may_or_must = 0
        # may_or_must[:, 0] = 1
    else:
        raise NotImplementedError('unrecognized task name!')
    # num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=dfa_specs.DFASPECS['dfa_v3'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'direction': direction,
                     'may_or_must': may_or_must,
                     'gen_vectors': dfa_probing.array_cat(gen_vectors, 2),
                     'kill_vectors': dfa_probing.array_cat(kill_vectors, 2),
                     'cfg_edges': cfg_edges,
                     # 'cfg_backward': cfg_edges_backward
                 })
    for time_idx in range(0, len(trace_list)):
    # for time_idx in range(0, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': dfa_probing.array_cat(trace_list[time_idx], 2)
                     })

    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': dfa_probing.array_cat(trace_list[-1], 2)})
    # expected_nb_nodes = dfa_sample_loader.max_num_pp + dfa_sample_loader.selected_num_ip
    expected_nb_cfg_edges = int(
        dfa_sample_loader.max_num_pp * dfa_sample_loader.cfg_edges_rate * 2) + dfa_sample_loader.max_num_pp
    edge_indices_dict, mask_dict = dfa_probing.finalize_for_dfa(probes=probes,
                                                                   expected_nb_nodes=dfa_sample_loader.max_num_pp,
                                                                   expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                                   expected_hint_len=dfa_sample_loader.expected_trace_len,  # used to be expected_hint_len
                                                                   num_ip=num_ip,
                                                                   full_trace_len=full_trace_len)
    return edge_indices_dict, mask_dict, probes

def dfa_plus(dfa_sample_loader: dfa_utils.SampleLoader,
           sample_id: str,
           task_name: str):
    '''sparse version'''
    trace_list, array_list, full_trace_len = dfa_sample_loader.load_a_sample(task_name=task_name,
                                                             sample_id=sample_id)
    gen_vectors, kill_vectors, cfg_edges = array_list
    # print(f'algo line 243, gen_vectors: {gen_vectors.shape}')
    # [num_pp, selected_num_ip ]
    num_ip = gen_vectors.shape[1]
    # may_or_must = np.zeros((num_ip, 2))
    if task_name == 'liveness':
        direction = 0
        may_or_must = 0
        # may_or_must[:, 0] = 1
    elif task_name == 'dominance':
        direction = 1
        may_or_must = 1
        # may_or_must[:, 1] = 1
    elif task_name == 'reachability':
        direction = 0
        may_or_must = 0
        # may_or_must[:, 0] = 1
    else:
        raise NotImplementedError('unrecognized task name!')
    # num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=dfa_specs.DFASPECS['dfa_v4'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'direction': direction,
                     'may_or_must': may_or_must,
                     'gen_vectors': dfa_probing.array_cat(gen_vectors, 2),
                     'kill_vectors': dfa_probing.array_cat(kill_vectors, 2),
                     'cfg_edges': cfg_edges,
                     # 'cfg_backward': cfg_edges_backward
                 })
    for time_idx in range(0, len(trace_list)):
    # for time_idx in range(0, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': dfa_probing.array_cat(trace_list[time_idx], 2)
                     })

    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': dfa_probing.array_cat(trace_list[-1], 2)})
    # expected_nb_nodes = dfa_sample_loader.max_num_pp + dfa_sample_loader.selected_num_ip
    expected_nb_cfg_edges = int(
        dfa_sample_loader.max_num_pp * dfa_sample_loader.cfg_edges_rate * 2) + dfa_sample_loader.max_num_pp
    edge_indices_dict, mask_dict = dfa_probing.finalize_for_dfa(probes=probes,
                                                                   expected_nb_nodes=dfa_sample_loader.max_num_pp,
                                                                   expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                                   expected_hint_len=dfa_sample_loader.expected_trace_len,  # used to be expected_hint_len
                                                                   num_ip=num_ip,
                                                                   full_trace_len=full_trace_len)
    return edge_indices_dict, mask_dict, probes