import numpy as np

from clrs._src import probing
from clrs._src import specs

_ProbesDict = probing.ProbesDict
_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type

_Array = np.ndarray


def dfa_finalize(probes: _ProbesDict,
                 expected_nb_nodes: int,
                 expected_nb_cfg_edges: int,
                 # expected_nb_gkt_edge: int,
                 expected_hint_len: int):
    """Finalizes a `ProbesDict` by stacking/squeezing `data` field."""
    padding_node_idx = expected_nb_nodes - 1
    nb_nodes = None
    nb_cfg_edges = None
    cfg_indices_padded = None
    hint_len = None
    # print(f'dfa_probing line 25 expected_nb_nodes = {expected_nb_nodes}')
    trace_h_padded = None
    for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
        for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            for name in probes[stage][loc]:
                if isinstance(probes[stage][loc][name]['data'], _Array):
                    raise probing.ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
                if name == 'trace_h':
                    hint_len = len(probes[stage][loc][name]['data'])
                    stacked_trace_h = np.stack(probes[stage][loc][name]['data'])
                    # [t, n, num_ip]
                    if nb_nodes is None:
                        nb_nodes = stacked_trace_h.shape[1]
                    else:
                        assert nb_nodes == stacked_trace_h.shape[1]
                    trace_h_mask_padding = np.repeat(
                        np.expand_dims(specs.OutputClass.MASKED * np.ones(
                            (expected_nb_nodes - nb_nodes, stacked_trace_h.shape[2])),
                                       axis=0),
                        repeats=hint_len,
                        axis=0)
                    #   [t, N-n, nb_ip]
                    trace_h_padded = np.concatenate([stacked_trace_h, trace_h_mask_padding],
                                                    axis=1)
                    #   [t, N, nb_ip]
                    # print(
                    #     f'dfa_probing line 52, stacked_trace_h: {stacked_trace_h.shape}; trace_h_padded: {trace_h_padded.shape}')
                else:
                    assert len(probes[stage][loc][name]['data']) == 1
                    old_data = probes[stage][loc][name]['data'][0]
                    # if name in ['direction', 'may_or_must']:
                    if name == 'direction':
                        probes[stage][loc][name]['data'] = np.repeat(old_data, repeats=expected_nb_cfg_edges)
                    elif name == 'cfg_edges':
                        nb_cfg_edges = old_data.shape[0]
                        cfg_indices_padding = padding_node_idx * np.ones((expected_nb_cfg_edges - nb_cfg_edges, 2),
                                                                         int)
                        cfg_indices_padded = np.concatenate([old_data[:, :2], cfg_indices_padding])
                        probes[stage][loc][name]['data'] = np.concatenate([old_data[:, -1],
                                                                           specs.OutputClass.MASKED * np.ones(
                                                                               expected_nb_cfg_edges - nb_cfg_edges)])
                    else:
                        assert name in ['gen_vectors', 'kill_vectors', 'trace_o']
                        if nb_nodes is None:
                            nb_nodes = old_data.shape[0]
                        else:
                            assert nb_nodes == old_data.shape[0]
                        probes[stage][loc][name]['data'] = np.concatenate([old_data,
                                                                           specs.OutputClass.MASKED * np.ones(
                                                                               (expected_nb_nodes - nb_nodes,
                                                                                old_data.shape[1]))],
                                                                          axis=0)
                        # [n, nb_ip] -> [N, nb_ip]

    padded_trace_o = probes[_Stage.OUTPUT][_Location.EDGE]['trace_o']['data']
    # [N, nb_ip]
    if hint_len < expected_hint_len:
        repeated_trace_o = np.repeat(np.expand_dims(padded_trace_o, 0),
                                     repeats=expected_hint_len - hint_len,
                                     axis=0)
        # [T-t, N, nb_ip]
        trace_h_padded = np.concatenate([trace_h_padded, repeated_trace_o], axis=0)
        # [T, N, nb_ip]
    probes[_Stage.HINT][_Location.EDGE]['trace_h']['data'] = trace_h_padded
    # tmp_data_trace_h = probes[_Stage.HINT][_Location.EDGE]['trace_h']['data']
    # print(f'dfa_probing line 113, trace_h: {tmp_data_trace_h.shape}')
    # for idx in range(tmp_data_trace_h.shape[0] - 1):
    #     print(f'if trace_{idx} the same with trace_{idx+1}? {np.array_equal(tmp_data_trace_h[idx], tmp_data_trace_h[idx + 1])}')
    # [T, E]
    edge_indices_dict = dict(cfg_bidirectional_indices_padded=cfg_indices_padded, )
    mask_dict = dict(nb_nodes=nb_nodes,
                     nb_cfg_edges=nb_cfg_edges,
                     hint_len=hint_len)
    print('dfa_probing line 126')
    # print(f'cfg_inidices_padded: \n{cfg_indices_padded}')
    # print(cfg_indices_padded) # checked~
    # print(f'gkt_indices_padded: \n{gkt_indices_padded}')
    # tmp = probes[specs.Stage.INPUT][specs.Location.EDGE]['gen']['data']
    # print(f'gen_padded: \n{tmp}')
    # if 'kill' in probes[specs.Stage.INPUT][specs.Location.EDGE]:
    #     tmp = probes[specs.Stage.INPUT][specs.Location.EDGE]['kill']['data']
    #     print(f'kill_padded: \n{tmp}')
    # print(gkt_indices_padded)   # checked~
    print(f'nb_nodes: {nb_nodes}; nb_cfg_edges: {nb_cfg_edges}; hint_len: {hint_len}')
    return edge_indices_dict, mask_dict
