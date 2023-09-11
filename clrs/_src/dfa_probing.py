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

    gkt_padding = None
    trace_h_node_padded = None
    for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
        for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            for name in probes[stage][loc]:
                if isinstance(probes[stage][loc][name]['data'], _Array):
                    raise probing.ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
                if name == 'trace_h':
                    hint_len = len(probes[stage][loc][name]['data'])
                    stacked_trace_h = np.stack(probes[stage][loc][name]['data'])
                    if nb_gkt_edges is None:
                        nb_gkt_edges = stacked_trace_h.shape[1]
                        gkt_padding = np.concatenate(
                            [padding_node_idx * np.ones((expected_nb_gkt_edge - nb_gkt_edges, 2),
                                                        int),
                             np.zeros((expected_nb_gkt_edge - nb_gkt_edges, 1),
                                      int)],
                            axis=1)

                    else:
                        assert nb_gkt_edges == stacked_trace_h.shape[1]
                        assert gkt_padding is not None
                    trace_h_node_padding = np.repeat(np.expand_dims(gkt_padding,
                                                                    axis=0),
                                                     hint_len)
                    trace_h_node_padded = np.concatenate([stacked_trace_h, trace_h_node_padding],
                                                         axis=0)
                    #   [t, E, 3]


                else:
                    len(probes[stage][loc][name]['data']) == 1
                    old_data = probes[stage][loc][name]['data']
                    if name in ['pos', 'if_pp', 'if_ip']:
                        nb_nodes = old_data.shape[0]
                        if name == 'pos':
                            probes[stage][loc][name]['data'] = np.copy(
                                np.arange(expected_nb_nodes)) * 1.0 / expected_nb_nodes
                        else:
                            # if_pp / if_ip
                            probes[stage][loc][name]['data'] = np.concatenate([old_data,
                                                                               np.zeros(expected_nb_nodes - nb_nodes,
                                                                                        int)],
                                                                              axis=0)
                    elif name == 'cfg':
                        nb_cfg_edges = old_data.shape[0]
                        cfg_padding_indices = padding_node_idx * np.ones((expected_nb_cfg_edges - nb_cfg_edges, 2),
                                                                         int)
                        cfg_padding_zeros = np.zeros((expected_nb_cfg_edges - nb_cfg_edges, 1),
                                                     int)
                        cfg_padding = np.concatenate([cfg_padding_indices, cfg_padding_zeros],
                                                     axis=1)
                        probes[stage][loc][name]['data'] = np.concatenate([old_data, cfg_padding],
                                                                          axis=0)
                    else:
                        assert name in ['gen', 'kill', 'trace_i', 'trace_o']
                        if nb_gkt_edges is None:
                            nb_gkt_edges = old_data.shape[0]
                            gkt_padding = np.concatenate(
                                [padding_node_idx * np.ones((expected_nb_gkt_edge - nb_gkt_edges, 2),
                                                            int),
                                 np.zeros((expected_nb_gkt_edge - nb_gkt_edges, 1),
                                          int)],
                                axis=1)
                        else:
                            assert nb_gkt_edges == old_data.shape[0]
                            assert gkt_padding is not None
                        padded_gkt = np.concatenate([old_data, gkt_padding],
                                                    axis=0)
                        probes[stage][loc][name]['data'] = padded_gkt

    padded_trace_o = probes[_Stage.OUTPUT][_Location.EDGE]['trace_o']['data']
    # [expected_nb_gkt_edges, 3]
    repeated_trace_o = np.repeat(np.expand_dims(padded_trace_o, 0),
                                 repeats=expected_hint_len - hint_len,
                                 axis=0)
    probes[_Stage.HINT][_Location.EDGE]['trace_h']['data'] = np.concatenate([trace_h_node_padded,
                                                                             repeated_trace_o],
                                                                            axis=0)
    return nb_nodes, nb_cfg_edges, nb_gkt_edges, hint_len
