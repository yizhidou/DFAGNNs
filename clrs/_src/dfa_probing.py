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
                 expected_nb_gkt_edge: int,
                 expected_hint_len: int):
    """Finalizes a `ProbesDict` by stacking/squeezing `data` field."""
    padding_node_idx = expected_nb_nodes - 1
    nb_nodes = None
    nb_cfg_edges = None
    nb_gkt_edges = None
    hint_len = None

    gkt_indices_padded = None
    cfg_indices_padded = None
    gkt_indices_padding = None
    trace_h_padded = None
    for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
        for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            for name in probes[stage][loc]:
                if isinstance(probes[stage][loc][name]['data'], _Array):
                    raise probing.ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
                if name == 'trace_h':
                    hint_len = len(probes[stage][loc][name]['data'])
                    stacked_trace_h = np.stack([x[:, -1] for x in probes[stage][loc][name]['data']])
                    # [t, e]
                    if nb_gkt_edges is None:
                        nb_gkt_edges = stacked_trace_h.shape[1]
                    else:
                        assert nb_gkt_edges == stacked_trace_h.shape[1]
                    # trace_h_zero_padding = np.repeat(np.expand_dims(np.zeros(expected_nb_gkt_edge - nb_gkt_edges),
                    #                                                 axis=0),
                    #                                  repeats=hint_len,
                    #                                  axis=0)
                    trace_h_mask_padding = np.repeat(
                        np.expand_dims(specs.OutputClass.MASKED * np.ones(expected_nb_gkt_edge - nb_gkt_edges),
                                       axis=0),
                        repeats=hint_len,
                        axis=0)
                    #   [t, E-e]
                    trace_h_padded = np.concatenate([stacked_trace_h, trace_h_mask_padding],
                                                    axis=1)
                    #   [t, E]
                else:
                    len(probes[stage][loc][name]['data']) == 1
                    old_data = probes[stage][loc][name]['data'][0]
                    if name in ['pos', 'if_pp', 'if_ip']:
                        if nb_nodes is None:
                            nb_nodes = old_data.shape[0]
                        else:
                            assert nb_nodes == old_data.shape[0]
                        if name == 'pos':
                            probes[stage][loc][name]['data'] = np.copy(
                                np.arange(expected_nb_nodes)) * 1.0 / expected_nb_nodes
                        else:
                            # if_pp / if_ip
                            probes[stage][loc][name]['data'] = np.concatenate([old_data,
                                                                               # np.zeros(expected_nb_nodes - nb_nodes)],
                                                                               specs.OutputClass.MASKED * np.ones(
                                                                                   expected_nb_nodes - nb_nodes)],
                                                                              axis=0)
                    elif name == 'cfg':
                        nb_cfg_edges = old_data.shape[0]
                        cfg_indices_padding = padding_node_idx * np.ones((expected_nb_cfg_edges - nb_cfg_edges, 2),
                                                                         int)
                        cfg_indices_padded = np.concatenate([old_data, cfg_indices_padding])
                        probes[stage][loc][name]['data'] = np.concatenate([np.ones(nb_cfg_edges),
                                                                           # np.zeros(expected_nb_cfg_edges - nb_cfg_edges)])
                                                                           specs.OutputClass.MASKED * np.ones(
                                                                               expected_nb_cfg_edges - nb_cfg_edges)])
                    else:
                        assert name in ['gen', 'kill', 'trace_i', 'trace_o']
                        if nb_gkt_edges is None:
                            nb_gkt_edges = old_data.shape[0]
                            gkt_indices_padding = padding_node_idx * np.ones((expected_nb_gkt_edge - nb_gkt_edges, 2),
                                                                             int)
                        else:
                            assert nb_gkt_edges == old_data.shape[0]
                            assert gkt_indices_padding is not None
                        gkt_indices_padded = np.concatenate([old_data[:, :2],
                                                             gkt_indices_padding])
                        #   [E, 2]
                        probes[stage][loc][name]['data'] = np.concatenate([old_data[:, -1],
                                                                           # np.zeros(expected_nb_gkt_edge - nb_gkt_edges)])
                                                                           specs.OutputClass.MASKED * np.ones(
                                                                               expected_nb_gkt_edge - nb_gkt_edges)])
                # if name != 'trace_h':
                #     tmp_data = probes[stage][loc][name]['data']
                #     print(f'dfa_probing line 101, {name}: {tmp_data.shape}')
                #     print(repr(tmp_data))
    padded_trace_o = probes[_Stage.OUTPUT][_Location.EDGE]['trace_o']['data']
    # [E, ]
    repeated_trace_o = np.repeat(np.expand_dims(padded_trace_o, 0),
                                 repeats=expected_hint_len - hint_len,
                                 axis=0)
    # [T-t, E]
    probes[_Stage.HINT][_Location.EDGE]['trace_h']['data'] = np.concatenate([trace_h_padded,
                                                                             repeated_trace_o],
                                                                            axis=0)
    tmp_data = probes[_Stage.HINT][_Location.EDGE]['trace_h']['data']
    print(f'dfa_probing line 113, trace_h: {tmp_data.shape}')
    for idx in range(tmp_data.shape[0] - 1):
        print(f'if trace_{idx+1} the same with trace_{idx + 2}? {np.array_equal(tmp_data[idx], tmp_data[idx + 1])}')
    # [T, E]
    edge_indices_dict = dict(cfg_indices_padded=cfg_indices_padded,
                             gkt_indices_padded=gkt_indices_padded)
    mask_dict = dict(nb_nodes=nb_nodes,
                     nb_cfg_edges=nb_cfg_edges,
                     nb_gkt_edges=nb_gkt_edges,
                     hint_len=hint_len)
    print('dfa_probing line 117')
    print(f'cfg_inidices_padded: {cfg_indices_padded.shape}')
    print(cfg_indices_padded)
    print(f'gkt_indices_padded: {gkt_indices_padded.shape}')
    print(gkt_indices_padded)
    print(f'nb_nodes: {nb_nodes}; nb_cfg_edges: {nb_cfg_edges}; nb_gkt_edges: {nb_gkt_edges}; hint_len: {hint_len}')
    return edge_indices_dict, mask_dict
