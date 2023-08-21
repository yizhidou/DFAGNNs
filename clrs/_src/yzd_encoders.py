from clrs._src import encoders
from clrs._src import specs
from clrs._src import yzd_probing
from typing import List, Union
import jax.numpy as jnp
import collections

_DataPoint = yzd_probing.DataPoint
_Trajactory = List[_DataPoint]
_ArraySparse = yzd_probing._ArraySparse
_ArrayDense = yzd_probing._ArrayDense
_Array = yzd_probing._Array
_Location = specs.Location
_Type = specs.Type

AdjSparse = collections.namedtuple('AdjSparse', ['row_indices', 'col_indices', 'nb_edges_each_graph'])


def func(dense_inputs: _Trajactory,
         sparse_inputs: _Trajactory,
         dense_hints: Union[_Trajactory, None] = None,
         sparse_hints: Union[_Trajactory, None] = None):
    result = {}
    nb_nodes_entire_batch = None
    for dp in dense_inputs:
        assert isinstance(dp.data, _ArrayDense)
        if dp.name == 'pos':
            if nb_nodes_entire_batch:
                assert dp.data.shape[0] == nb_nodes_entire_batch
            else:
                nb_nodes_entire_batch = dp.data.shape[0]
            result['pos_content'] = jnp.array(dp.data)
        if dp.name == 'if_pp':
            if nb_nodes_entire_batch:
                assert dp.data.shape[0] == nb_nodes_entire_batch
            else:
                nb_nodes_entire_batch = dp.data.shape[0]
            result['if_pp_content'] = jnp.array(dp.data)
        if dp.name == 'if_ip':
            if nb_nodes_entire_batch:
                assert dp.data.shape[0] == nb_nodes_entire_batch
            else:
                nb_nodes_entire_batch = dp.data.shape[0]
            result['if_ip_content'] = jnp.array(dp.data)

    kill_edges = None
    for dp in sparse_inputs:
        assert isinstance(dp.data, _ArraySparse)
        if dp.name == 'cfg_sparse':
            result['cfg_edges'] = jnp.array(dp.data.edge_indices_with_optional_content)
            result['nb_cfg_edges'] = jnp.array(dp.data.nb_edges)
        if dp.name == 'gen_sparse':
            gen_edges = jnp.array(dp.data.edge_indices_with_optional_content[:, :-1])
            nb_gen_edges = jnp.array(dp.data.nb_edges)
            result['gen_content'] = jnp.array(dp.data.edge_indices_with_optional_content[:, -1])
        if dp.name == 'kill_sparse':
            kill_edges = jnp.array(dp.data.edge_indices_with_optional_content[:, :-1])
            nb_kill_edges = jnp.array(dp.data.nb_edges)
            result['kill_content'] = jnp.array(dp.data.edge_indices_with_optional_content[:, -1])
        if dp.name == 'trace_i_sparse':
            trace_i_edges = jnp.array(dp.data.edge_indices_with_optional_content[:, :-1])
            result['trace_i_content'] = jnp.array(dp.data.edge_indices_with_optional_content[:, -1])
    if kill_edges:
        assert jnp.array_equal(gen_edges, kill_edges)
        assert jnp.array_equal(nb_gen_edges, nb_kill_edges)
    assert jnp.array_equal(gen_edges, trace_i_edges)
    result['kg_edges'] = gen_edges
    result['nb_kg_edges'] = nb_gen_edges

    if sparse_hints:
        assert len(sparse_hints) == 1
        assert sparse_hints[0].name == 'trace_h_sparse'
        result['trace_h_edges'] = jnp.array(sparse_hints[0].data.edge_indices_with_optional_content[:, :-1])
        result['nb_trace_h_edges'] = jnp.sum(jnp.array(sparse_hints[0].data.nb_edges),
                                             axis=-1)


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
