from clrs._src import encoders
from clrs._src import specs
from clrs._src import yzd_probing
from typing import List
import jax.numpy as jnp
import collections

_DataPoint = yzd_probing.DataPoint
_ArraySparse = yzd_probing._ArraySparse
_ArrayDense = yzd_probing._ArrayDense
_Array = yzd_probing._Array
_Location = specs.Location
_Type = specs.Type

AdjSparse = collections.namedtuple('AdjSparse', ['row_indices', 'col_indices', 'nb_edges_each_graph'])

def accum_adj_and_node_edge_fts(dp: _DataPoint, adj_list: List[_ArrayDense]):
    if dp.name == 'cfg':
        assert isinstance(dp.data, _ArraySparse)



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
