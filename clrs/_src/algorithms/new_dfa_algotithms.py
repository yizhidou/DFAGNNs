import numpy as np

from clrs._src import dfa_utils
from clrs._src import dfa_specs, specs
from clrs._src import probing, dfa_probing


def liveness(dfa_sample_loader: dfa_utils.SampleLoader,
             sample_id: str):
    '''sparse version'''
    trace_list, array_list, if_pp, if_ip = dfa_sample_loader.load_a_sample(task_name='liveness',
                                                                           sample_id=sample_id)
    cfg_sparse, gen_sparse, kill_sparse = array_list
    num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=dfa_specs.DFASPECS['dfa_liveness'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'pos': np.copy(np.arange(num_nodes)) * 1.0 / num_nodes,
                     'cfg': cfg_sparse,
                     'gen': gen_sparse,
                     'kill': kill_sparse,
                     'if_pp': if_pp,
                     'if_ip': if_ip,
                     # 'trace_i': trace_list[0]
                 })
    # for time_idx in range(1, len(trace_list) - 1):
    for time_idx in range(0, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': trace_list[time_idx],
                         # 'time': time_idx
                     })

    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': trace_list[-1]})
    expected_nb_nodes = dfa_sample_loader.max_num_pp + dfa_sample_loader.selected_num_ip
    expected_nb_cfg_edges = int(dfa_sample_loader.max_num_pp * dfa_sample_loader.cfg_edges_rate)
    expected_nb_gkt_edge = dfa_sample_loader.max_num_pp * dfa_sample_loader.selected_num_ip
    edge_indices_dict, mask_dict = dfa_probing.finalize_for_ldr(probes=probes,
                                                                    expected_nb_nodes=expected_nb_nodes,
                                                                    expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                                    expected_nb_gkt_edge=expected_nb_gkt_edge,
                                                                    expected_hint_len=dfa_sample_loader.expected_hint_len)
    return edge_indices_dict, mask_dict, probes


def dominance(dfa_sample_loader: dfa_utils.SampleLoader,
              sample_id: str):
    # assert dfa_sample_loader.if_sparse
    trace_list, array_list, if_pp, if_ip = dfa_sample_loader.load_a_sample(task_name='dominance',
                                                                           sample_id=sample_id)
    cfg_sparse, gen_sparse = array_list
    num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=dfa_specs.DFASPECS['dfa_dominance'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'pos': np.copy(np.arange(num_nodes)) * 1.0 / num_nodes,
                     'cfg': cfg_sparse,
                     'gen': gen_sparse,
                     'if_pp': if_pp,
                     'if_ip': if_ip,
                     # 'trace_i': trace_list[0]
                 })
    # for time_idx in range(1, len(trace_list) - 1):
    for time_idx in range(0, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': trace_list[time_idx],
                         # 'time': time_idx
                     }
                     )
    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': trace_list[-1]})
    expected_nb_nodes = dfa_sample_loader.max_num_pp
    expected_nb_cfg_edges = int(dfa_sample_loader.max_num_pp * dfa_sample_loader.cfg_edges_rate)
    expected_nb_gkt_edge = dfa_sample_loader.max_num_pp * dfa_sample_loader.selected_num_ip

    edge_indices_dict, mask_dict = dfa_probing.finalize_for_ldr(probes=probes,
                                                                    expected_nb_nodes=expected_nb_nodes,
                                                                    expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                                    expected_nb_gkt_edge=expected_nb_gkt_edge,
                                                                    expected_hint_len=dfa_sample_loader.expected_hint_len)
    return edge_indices_dict, mask_dict, probes


def reachability(dfa_sample_loader: dfa_utils.SampleLoader,
                 sample_id: str):
    # assert dfa_sample_loader.if_sparse
    trace_list, array_list, if_pp, if_ip = dfa_sample_loader.load_a_sample(task_name='reachability',
                                                                           sample_id=sample_id)
    cfg_sparse, gen_sparse = array_list
    num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=dfa_specs.DFASPECS['reachability'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'pos': np.copy(np.arange(num_nodes)) * 1.0 / num_nodes,
                     'cfg': cfg_sparse,
                     'gen': gen_sparse,
                     'if_pp': if_pp,
                     'if_ip': if_ip,
                     # 'trace_i': trace_list[0]
                 })
    # for time_idx in range(1, len(trace_list) - 1):
    for time_idx in range(0, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': trace_list[time_idx],
                         # 'time': time_idx
                     }
                     )
    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': trace_list[-1]})
    expected_nb_nodes = dfa_sample_loader.max_num_pp
    expected_nb_cfg_edges = int(dfa_sample_loader.max_num_pp * dfa_sample_loader.cfg_edges_rate)
    expected_nb_gkt_edge = dfa_sample_loader.max_num_pp * dfa_sample_loader.selected_num_ip

    edge_indices_dict, mask_dict = dfa_probing.finalize_for_ldr(probes=probes,
                                                                    expected_nb_nodes=expected_nb_nodes,
                                                                    expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                                    expected_nb_gkt_edge=expected_nb_gkt_edge,
                                                                    expected_hint_len=dfa_sample_loader.expected_hint_len)
    return edge_indices_dict, mask_dict, probes


def dfa(dfa_sample_loader: dfa_utils.SampleLoader,
        sample_id: str,
        task_name: str):
    '''sparse version'''
    trace_list, array_list = dfa_sample_loader.load_a_sample(task_name=task_name,
                                                             sample_id=sample_id)
    gen_vectors, kill_vectors, cfg_edges = array_list
    if task_name == 'liveness':
        direction = np.zeros(1)
        # may_or_must = np.ones()
    elif task_name == 'dominance':
        direction = np.ones(1)
        # may_or_must = np.ones()
    elif task_name == 'reachability':
        direction = np.zeros(1)
        # may_or_must = np.ones()
    else:
        raise NotImplementedError('unrecognized task name!')
    # num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=dfa_specs.DFASPECS['dfa'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'direction': direction,
                     # 'may_or_must': may_or_must,
                     'gen_vectors': gen_vectors,
                     'kill_vectors': kill_vectors,
                     'cfg_edges': cfg_edges,
                     # 'cfg_backward': cfg_edges_backward
                 })
    # for time_idx in range(1, len(trace_list) - 1):
    for time_idx in range(0, len(trace_list)):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': trace_list[time_idx]
                     })

    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o': trace_list[-1]})
    # expected_nb_nodes = dfa_sample_loader.max_num_pp + dfa_sample_loader.selected_num_ip
    expected_nb_cfg_edges = int(dfa_sample_loader.max_num_pp * dfa_sample_loader.cfg_edges_rate * 2)
    edge_indices_dict, mask_dict = dfa_probing.finalize_for_dfa(probes=probes,
                                                                    expected_nb_nodes=dfa_sample_loader.max_num_pp,
                                                                    expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                                    expected_hint_len=dfa_sample_loader.expected_trace_len)
    return edge_indices_dict, mask_dict, probes
