import numpy as np

from clrs._src import probing
from clrs._src import specs

_ProbesDict = probing.ProbesDict
_Location = specs.Location
_Stage = specs.Stage
_Type = specs.Type

_Array = np.ndarray


def finalize(probes: _ProbesDict,
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
    for stage in [_Stage.INPUT, _Stage.OUTPUT, _Stage.HINT]:
        for loc in [_Location.NODE, _Location.EDGE, _Location.GRAPH]:
            for name in probes[stage][loc]:
                if isinstance(probes[stage][loc][name]['data'], _Array):
                    raise probing.ProbeError('Attemping to re-finalize a finalized `ProbesDict`.')
                if stage == _Stage.HINT:
                    # Hints are provided for each timestep. Stack them here.
                    probes[stage][loc][name]['data'] = np.stack(
                        probes[stage][loc][name]['data'])
                else:
                    # Only one instance of input/output exist. Remove leading axis.
                    probes[stage][loc][name]['data'] = np.squeeze(
                        np.array(probes[stage][loc][name]['data']))
                if name == 'trace_h':
                    hint_len = len(probes[stage][loc][name]['data'])
                    stacked_trace_h = np.stack(probes[stage][loc][name]['data'])
                    if nb_gkt_edges is None:
                        nb_gkt_edges = stacked_trace_h.shape[1]
                        trace_h_node_padding = np.repeat(np.expand_dims(gkt_padding,
                                                                        axis=0),
                                                         hint_len)
                        trace_h_node_padded = np.concatenate([stacked_trace_h, trace_h_node_padding],
                                                             axis=0)
                        #   [t, E, 3]
                    else:
                        assert nb_gkt_edges == stacked_trace_h.shape[1]


                else:
                    len(probes[stage][loc][name]['data']) == 1
                    old_data = probes[stage][loc][name]['data']
                    if name == 'pos':
                        nb_nodes = old_data.shape[0]
                        probes[stage][loc][name]['data'] = np.copy(
                            np.arange(expected_nb_nodes)) * 1.0 / expected_nb_nodes
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
