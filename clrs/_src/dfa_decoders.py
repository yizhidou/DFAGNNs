from typing import Dict, Optional
from clrs._src import decoders
from clrs._src import specs
from clrs._src import yzd_probing

import chex
import haiku as hk
import jax
import jax.numpy as jnp

_chex_Array = chex.Array
_DataPoint = yzd_probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type


def decode_fts(decoders,
               h_t: _chex_Array,
               gkt_edge_fts: _chex_Array,
               gkt_edge_indices: _chex_Array):
    decoded_trace_o = _decode_gkt_edge_fts(decoders=decoders['trace_o'],
                                           h_t=h_t,
                                           gkt_edge_fts=gkt_edge_fts,
                                           gkt_edge_indices=gkt_edge_indices)
    decoded_trace_h = None
    if 'trace_h' in decoders:
        decoded_trace_h = _decode_gkt_edge_fts(decoders=decoders['trace_h'],
                                           h_t=h_t,
                                           gkt_edge_fts=gkt_edge_fts,
                                           gkt_edge_indices=gkt_edge_indices)
    return decoded_trace_h, decoded_trace_o


def _decode_gkt_edge_fts(decoders,
                         h_t: _chex_Array,
                         gkt_edge_fts: _chex_Array,
                         gkt_edge_indices: _chex_Array) -> _chex_Array:
    """Decodes edge features."""

    pred_1 = decoders[0](h_t)
    pred_2 = decoders[1](h_t)
    pred_e = decoders[2](gkt_edge_fts)
    gkt_edges_row_indices = gkt_edge_indices[:, 0]
    gkt_edges_col_indices = gkt_edge_indices[:, 1]
    preds = pred_1[gkt_edges_row_indices] + pred_2[gkt_edges_col_indices] + pred_e
    preds = jnp.squeeze(preds, axis=-1)
    return preds
