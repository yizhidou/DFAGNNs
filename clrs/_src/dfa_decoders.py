from clrs._src import specs
from clrs._src import dfa_utils

import chex
import jax.numpy as jnp

_chex_Array = chex.Array
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
                         h_t: _chex_Array,  # [B, N, 3*hidden_dim]
                         gkt_edge_fts: _chex_Array,  # [B, E_gkt, hidden_dim]
                         gkt_edge_indices: _chex_Array) -> _chex_Array:
    """Decodes edge features."""

    pred_1 = decoders[0](h_t)  # [B, N, 1]
    pred_2 = decoders[1](h_t)  # [B, N, 1]
    pred_e = decoders[2](gkt_edge_fts)  # [B, E_gkt, 1]
    gkt_edges_row_indices = gkt_edge_indices[..., 0]  # [B, E_gkt,]
    gkt_edges_col_indices = gkt_edge_indices[..., 1]  # [B, E_gkt,]
    preds = jnp.take_along_axis(arr=pred_1,
                                indices=dfa_utils.dim_expand_to(gkt_edges_row_indices, pred_1),
                                axis=1) + \
            jnp.take_along_axis(arr=pred_2,
                                indices=dfa_utils.dim_expand_to(gkt_edges_col_indices, pred_2),
                                axis=1) + \
            pred_e  # [B, E_gkt, 1]
    # print(f'dfa_decoders line 54, pred_1: {pred_1.shape}; pred_e: {pred_e.shape}; preds: {preds.shape}')    # checked
    preds = jnp.squeeze(preds, axis=-1)  # [B, E_gkt]
    return preds
