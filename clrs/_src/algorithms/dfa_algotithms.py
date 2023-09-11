import numpy as np

from clrs._src import yzd_utils
from clrs._src import yzd_specs, specs
from clrs._src import probing, dfa_probing


def dfa_liveness(dfa_sample_loader: yzd_utils.SampleLoader,
                 sample_id: str):
    '''sparse version'''
    assert not dfa_sample_loader.if_sparse
    # max_hint_len = yzd_sample_loader.max_iteration - 1
    trace_list, array_list, if_pp, if_ip = dfa_sample_loader.load_a_sample(task_name='yzd_liveness',
                                                                           sample_id=sample_id)
    cfg_sparse, gen_sparse, kill_sparse = array_list
    num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=yzd_specs.YZDSPECS['yzd_liveness'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'pos': np.copy(np.arange(num_nodes)) * 1.0 / num_nodes,
                     'cfg': cfg_sparse,
                     'gen': gen_sparse,
                     'kill': kill_sparse,
                     'trace_i': trace_list[0]})

    for time_idx in range(1, len(trace_list) - 1):
        probing.push(probes,
                     specs.Stage.HINT,
                     next_probe={
                         'trace_h': trace_list[time_idx],
                         # 'time': time_idx
                     })

    probing.push(probes,
                 specs.Stage.OUTPUT,
                 next_probe={'trace_o_sparse': trace_list[-1]})
    expected_nb_nodes = dfa_sample_loader.max_num_pp + dfa_sample_loader.selected_num_ip
    expected_nb_cfg_edges = dfa_sample_loader.max_num_pp * dfa_sample_loader.gkt_edges_rate
    expected_nb_gkt_edge = dfa_sample_loader.max_num_pp * dfa_sample_loader.selected_num_ip
    nb_nodes, nb_cfg_edges, nb_gkt_edges, hint_len = dfa_probing.dfa_finalize(probes=probes,
                                                                              expected_nb_nodes=expected_nb_nodes,
                                                                              expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                                              expected_nb_gkt_edge=expected_nb_gkt_edge,
                                                                              expected_hint_len=dfa_sample_loader.expected_hint_len)
    return nb_nodes, nb_cfg_edges, nb_gkt_edges, hint_len, probes


def dfa_dominance(dfa_sample_loader: yzd_utils.SampleLoader,
                  sample_id: str):
    assert dfa_sample_loader.if_sparse
    trace_list, array_list, if_pp, if_ip = dfa_sample_loader.load_a_sample(task_name='yzd_dominance',
                                                                           sample_id=sample_id)
    cfg_sparse, gen_sparse = array_list
    num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=yzd_specs.YZDSPECS['yzd_dominance'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'pos': np.copy(np.arange(num_nodes)) * 1.0 / num_nodes,
                     'cfg': cfg_sparse,
                     'gen': gen_sparse,
                     'trace_i': trace_list[0]
                 })
    for time_idx in range(1, len(trace_list) - 1):
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
    expected_nb_cfg_edges = dfa_sample_loader.max_num_pp * dfa_sample_loader.gkt_edges_rate
    expected_nb_gkt_edge = dfa_sample_loader.max_num_pp * dfa_sample_loader.selected_num_ip

    nb_nodes, nb_cfg_edges, nb_gkt_edges, hint_len = dfa_probing.dfa_finalize(probes=probes,
                                                                              expected_nb_nodes=expected_nb_nodes,
                                                                              expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                                              expected_nb_gkt_edge=expected_nb_gkt_edge,
                                                                              expected_hint_len=dfa_sample_loader.expected_hint_len)
    return nb_nodes, nb_cfg_edges, nb_gkt_edges, hint_len, probes


def dfa_reachability(dfa_sample_loader: yzd_utils.SampleLoader,
                     sample_id: str):
    assert dfa_sample_loader.if_sparse
    trace_list, array_list, if_pp, if_ip = dfa_sample_loader.load_a_sample(task_name='yzd_reachability',
                                                                           sample_id=sample_id)
    cfg_sparse, gen_sparse = array_list
    num_nodes = if_pp.shape[0]
    probes = probing.initialize(spec=yzd_specs.YZDSPECS['yzd_reachability'])
    probing.push(probes,
                 specs.Stage.INPUT,
                 next_probe={
                     'pos': np.copy(np.arange(num_nodes)) * 1.0 / num_nodes,
                     'cfg': cfg_sparse,
                     'gen': gen_sparse,
                     'trace_i': trace_list[0]
                 })
    for time_idx in range(1, len(trace_list) - 1):
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
    expected_nb_cfg_edges = dfa_sample_loader.max_num_pp * dfa_sample_loader.gkt_edges_rate
    expected_nb_gkt_edge = dfa_sample_loader.max_num_pp * dfa_sample_loader.selected_num_ip

    nb_nodes, nb_cfg_edges, nb_gkt_edges, hint_len = dfa_probing.dfa_finalize(probes=probes,
                                                                              expected_nb_nodes=expected_nb_nodes,
                                                                              expected_nb_cfg_edges=expected_nb_cfg_edges,
                                                                              expected_nb_gkt_edge=expected_nb_gkt_edge,
                                                                              expected_hint_len=dfa_sample_loader.expected_hint_len)
    return nb_nodes, nb_cfg_edges, nb_gkt_edges, hint_len, probes
