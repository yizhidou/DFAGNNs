import numpy as np

from clrs._src import probing
from clrs._src import specs

_ProbesDict = probing.ProbesDict
_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type

_Array = np.ndarray


def finalize_for_dfa(probes: _ProbesDict,
                        expected_nb_nodes: int,
                        expected_nb_cfg_edges: int,
                        expected_hint_len: int,
                        num_ip: int,
                        full_trace_len: int):
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
                    # [t, n, num_ip, 2]
                    if nb_nodes is None:
                        nb_nodes = stacked_trace_h.shape[1]
                    else:
                        assert nb_nodes == stacked_trace_h.shape[1]
                    trace_h_mask_padding_content = np.zeros((expected_nb_nodes - nb_nodes, stacked_trace_h.shape[2], 2))
                    # [N-n, num_ip, 2]
                    trace_h_mask_padding_content[..., 0] = specs.OutputClass.MASKED
                    trace_h_mask_padding = np.repeat(
                        np.expand_dims(trace_h_mask_padding_content, axis=0),
                        repeats=hint_len,
                        axis=0)
                    #   [t, N-n, num_ip, 2]
                    trace_h_padded = np.concatenate([stacked_trace_h, trace_h_mask_padding],
                                                    axis=1)
                    #   [t, N, num_ip, 2]
                else:
                    assert len(probes[stage][loc][name]['data']) == 1
                    old_data = probes[stage][loc][name]['data'][0]
                    # if name in ['direction', 'may_or_must']:
                    if name == 'direction':
                        if loc == _Location.EDGE:
                            direction_data = np.zeros((expected_nb_cfg_edges, 2))
                            direction_data[:, old_data] = 1
                            # [E, 2]
                        else:
                            assert loc == _Location.NODE
                            direction_data = np.zeros((num_ip, 2))
                            direction_data[:, old_data] = 1
                            # [m, 2]
                            direction_data = np.expand_dims(direction_data, axis=0)
                            # [1, m ,2]
                            direction_data = np.repeat(direction_data, axis=0, repeats=expected_nb_nodes)
                            # [N, m, 2]
                        probes[stage][loc][name]['data'] = direction_data
                    elif name == 'may_or_must':
                        if loc == _Location.NODE:
                            mm_data = np.zeros((num_ip, 2))
                            mm_data[:, old_data] = 1
                            # [m, 2]
                            mm_data = np.expand_dims(mm_data, axis=0)
                            #   [1, m, 2]
                            mm_data = np.repeat(mm_data, axis=0, repeats=expected_nb_nodes)
                            #   [N, m, 2]
                            # print(f'dfa_probing line 300, mm_data: {mm_data.shape}')
                        else:
                            assert loc == _Location.EDGE
                            mm_data = np.zeros((expected_nb_cfg_edges, 2))
                            # [E, 2]
                            mm_data[:, old_data] = 1
                            # [E, 2]
                        probes[stage][loc][name]['data'] = mm_data
                    elif name == 'cfg_edges':
                        # add self-loops to every node
                        self_loops_indices = np.expand_dims(np.arange(expected_nb_nodes), axis=-1) * np.ones(
                            (expected_nb_nodes, 2), dtype=int)
                        # [N, 2]
                        nb_cfg_edges = old_data.shape[0] + expected_nb_nodes
                        cfg_indices_padding = padding_node_idx * np.ones((expected_nb_cfg_edges - nb_cfg_edges, 2),
                                                                         int)
                        cfg_indices_padded = np.concatenate([old_data[:, :2],
                                                             self_loops_indices,
                                                             cfg_indices_padding])
                        # [E, 2]
                        self_loops_content = 2 * np.ones(expected_nb_nodes, dtype=int)
                        # [N, ]
                        cfg_edges = np.concatenate([old_data[:, 2], self_loops_content])
                        # [e, ]
                        cfg_edges_data = np.zeros((nb_cfg_edges, 3))
                        for i in range(nb_cfg_edges):
                            # try:
                            cfg_edges_data[i, cfg_edges[i]] = 1
                            # except IndexError:
                            #     print(f'IndexError happens! i = {i}; old_data[i] = {old_data[i]}')
                            #     exit(666)
                        # [e, 3]
                        cfg_edges_data_padding = np.zeros((expected_nb_cfg_edges - nb_cfg_edges, 3))
                        cfg_edges_data_padding[:, 2] = 1
                        # [E-e, 3]
                        # The padding is self-loops of the last node
                        probes[stage][loc][name]['data'] = np.concatenate([cfg_edges_data,
                                                                           cfg_edges_data_padding],
                                                                          axis=0)
                        # [E, 3]
                    else:
                        assert name in ['gen_vectors', 'kill_vectors', 'trace_o']
                        # print(f'new_dfa_probing line 194, {name}: {old_data.shape}')
                        if nb_nodes is None:
                            nb_nodes = old_data.shape[0]
                        else:
                            assert nb_nodes == old_data.shape[0]
                        data_padding = np.zeros((expected_nb_nodes - nb_nodes, old_data.shape[1], 2))
                        # [N-n, m, 2]
                        data_padding[..., 0] = specs.OutputClass.MASKED
                        # print(f'dfa_probing line 314, {name}. old_data: {old_data.shape}; data_padding: {data_padding.shape}')
                        probes[stage][loc][name]['data'] = np.concatenate([old_data, data_padding],
                                                                          axis=0)

                        # [N, nb_ip, 2]
                    # print('new_dfa_probing line 205, {}: {}'.format(name, probes[stage][loc][name]['data'].shape))
    padded_trace_o = probes[_Stage.OUTPUT][_Location.NODE]['trace_o']['data']
    # [N, nb_ip, 2]
    if hint_len < expected_hint_len:
        repeated_trace_o = np.repeat(np.expand_dims(padded_trace_o, 0),
                                     repeats=expected_hint_len - hint_len,
                                     axis=0)
        # [T-t, N, nb_ip, 2]
        trace_h_padded = np.concatenate([trace_h_padded, repeated_trace_o], axis=0)
        # [T, N, nb_ip, 2]
    probes[_Stage.HINT][_Location.NODE]['trace_h']['data'] = trace_h_padded
    # tmp_data_trace_h = probes[_Stage.HINT][_Location.EDGE]['trace_h']['data']
    # print(f'dfa_probing line 113, trace_h: {tmp_data_trace_h.shape}')
    # for idx in range(tmp_data_trace_h.shape[0] - 1):
    #     print(f'if trace_{idx} the same with trace_{idx+1}? {np.array_equal(tmp_data_trace_h[idx], tmp_data_trace_h[idx + 1])}')
    # [T, E]
    edge_indices_dict = dict(cfg_indices_padded=cfg_indices_padded, )
    mask_dict = dict(nb_nodes=nb_nodes,
                     nb_cfg_edges=nb_cfg_edges,
                     hint_len=hint_len,
                     full_trace_len=full_trace_len)
    # print('dfa_probing line 226')
    # print(f'cfg_inidices_padded: {cfg_indices_padded.shape} \n{cfg_indices_padded}')
    # tmp = probes[specs.Stage.INPUT][specs.Location.NODE]['gen_vectors']['data']
    # print(f'gen_vectors: {tmp.shape} \n{tmp}')
    # print(f'nb_nodes: {nb_nodes}; nb_cfg_edges: {nb_cfg_edges}; hint_len: {hint_len}')
    return edge_indices_dict, mask_dict


def array_cat(A, num_cat):
    assert num_cat > 0 and len(A.shape) == 2
    probe = np.zeros(A.shape + (num_cat,))
    # print(f'dfa_probing line 344, probe: {probe.shape}')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            probe[i, j, A[i, j]] = 1
    return probe
    #   [N, m, num_cat]
