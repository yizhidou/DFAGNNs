from clrs._src import encoders
from clrs._src import specs
from clrs._src import yzd_probing
from typing import List, Union
import jax.numpy as jnp
import collections

_DataPoint = yzd_probing.DataPoint
_Trajactory = List[_DataPoint]
_ArraySparse = yzd_probing.ArraySparse
_ArrayDense = yzd_probing.ArrayDense
_Array = yzd_probing.Array
_Location = specs.Location
_Type = specs.Type

AdjSparse = collections.namedtuple('AdjSparse', ['row_indices', 'col_indices', 'nb_edges_each_graph'])


def func(input_NODE_dp_list: _Trajactory,
         input_EDGE_dp_list: _Trajactory,
         trace_h_i: Union[_DataPoint, None] = None):
    result = {}
    # nb_nodes_entire_batch = None
    for dp in input_NODE_dp_list:
        assert isinstance(dp.data, _ArrayDense)
        if dp.name == 'pos':
            result['pos_content'] = jnp.array(dp.data)
        if dp.name == 'if_pp':
            result['if_pp_content'] = jnp.array(dp.data)
        if dp.name == 'if_ip':
            result['if_ip_content'] = jnp.array(dp.data)

    kill_edges = None
    nb_nodes = None
    for dp in input_EDGE_dp_list:
        assert isinstance(dp.data, _ArraySparse)
        if dp.name == 'cfg_sparse':
            result['cfg_edges'] = jnp.array(dp.data.edges_with_optional_content)
            result['nb_cfg_edges'] = jnp.array(dp.data.nb_edges)
            # if not nb_nodes:
            #     nb_nodes = jnp.array(dp.data.nb_nodes)
            # else:
            #     assert nb_nodes_entire_batch == jnp.sum(nb_nodes).item()
            result['nb_nodes'] = nb_nodes
        if dp.name == 'gen_sparse':
            result['kgt_edges'] = jnp.array(dp.data.edges_with_optional_content[:, :-1])
            result['gen_content'] = jnp.array(dp.data.edges_with_optional_content[:, -1])
            result['nb_gkt_edges'] = jnp.array(dp.data.nb_edges)
        if dp.name == 'kill_sparse':
            result['kill_content'] = jnp.array(dp.data.edges_with_optional_content[:, -1])
        if dp.name == 'trace_i_sparse':
            result['trace_i_content'] = jnp.array(dp.data.edges_with_optional_content[:, -1])
    result['trace_h_i_content'] = jnp.array(trace_h_i.data.edges_with_optional_content[:, -1])
    return result


def get_gen_kill_edges(sparse_inputs: List[_DataPoint]):
    for dp in sparse_inputs:
        assert isinstance(dp.data, _ArraySparse)
        if dp.name == 'gen_sparse':
            gen_sparse = dp.data
        if dp.name == 'kill_sparse':
            kill_sparse = dp.data
    return gen_sparse, kill_sparse


def accum_adj_mat_yzd(dp: _DataPoint) -> _Array:
    """Accumulates adjacency matrix. (sparse version)"""

    if dp.name == 'cfg':
        assert isinstance(dp.data, yzd_probing._ArraySparse)
        adj_sparse = AdjSparse(row_indices=dp.data.edges[:, 0],
                               col_indices=dp.data.edges[:, 1],
                               nb_edges_each_graph=dp.data.nb_edges)

        return adj_sparse  # pytype: disable=attribute-error  # numpy-scalars


def accum_edge_fts_yzd(encoders, dp: _DataPoint, edge_fts: _Array) -> _Array:
    """Encodes and accumulates edge features."""
    if dp.location == _Location.EDGE:
        assert dp.type_ == _Type.MASK
        encoding = encoders[0](dp.data)
        edge_fts += encoding
    return edge_fts
