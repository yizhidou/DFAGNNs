import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.ops

import abc
from typing import Any, Callable, List, Optional, Tuple, Union

from clrs._src import dfa_utils

_chex_Array = chex.Array
_Fn = Callable[..., Any]
BIG_NUMBER = 1e6
PROCESSOR_TAG = 'dfa_processor'


class DFAProcessor(hk.Module):
    """Processor abstract base class."""

    def __init__(self, name: str):
        if not name.endswith(PROCESSOR_TAG):
            name = name + '_' + PROCESSOR_TAG
        super().__init__(name=name)

    @abc.abstractmethod
    def __call__(self,
                 # hidden: _chex_Array,
                 *args,
                 **kwargs):
        """Processor inference step.

        Returns:
          Output of processor inference step as a 2-tuple of (node, edge)
          embeddings. The edge embeddings can be None.
        """
        pass


class GATSparse(DFAProcessor):
    def __init__(
            self,
            out_size: int,
            nb_heads: int,
            activation: Optional[_Fn] = jax.nn.relu,
            residual: bool = True,
            use_ln: bool = False,
            name: str = 'gat_sparse',
    ):
        super().__init__(name=name)
        self.out_size = out_size
        self.nb_heads = nb_heads
        if out_size % nb_heads != 0:
            raise ValueError('The number of attention heads must divide the width!')
        self.head_size = out_size // nb_heads
        self.activation = activation
        self.residual = residual
        self.use_ln = use_ln

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            node_fts: _chex_Array,  # [B, N, hidden_dim]
            gkt_edge_fts: _chex_Array,  # [B, E_gkt, hidden_dim]
            hidden: _chex_Array,  # [B, N, hidden_dim]
            cfg_indices_padded: _chex_Array,  # [B, E_cfg, 2]
            gkt_indices_padded: _chex_Array,  # [B, E_gkt, 2]
    ):
        """GAT inference step (sparse version by yzd)."""
        # print('in dfa_processor line 71')   # checked
        # print(  # [B, N, hidden_dim],       [B, E_gkt, hidden_dim],                 [B, N, hidden_dim],         [B, E_cfg, 2],                                  [B, E_gkt, 2]
        #     f'node_fts: {node_fts.shape};\ngkt_edge_fts: {gkt_edge_fts.shape};\nhidden: {hidden.shape};\ncfg_indices_padded: {cfg_indices_padded.shape};\ngkt_indices_padded: {gkt_indices_padded.shape}')
        nb_nodes_padded = node_fts.shape[-2]
        nb_cfg_edges_padded, nb_gkt_edges_padded = cfg_indices_padded.shape[-2], gkt_indices_padded.shape[-2]
        # print(  # checked
        #     f'dfa_processor line 76, \nnb_nodes_padded = {nb_nodes_padded}; \nnb_cfg_edges_padded = {nb_cfg_edges_padded}; \nnb_gkt_edges_padded = {nb_gkt_edges_padded}')
        cfg_source_indices = cfg_indices_padded[..., 0]  # [B, E_cfg]
        cfg_target_indices = cfg_indices_padded[..., 1]
        gkt_source_indices = gkt_indices_padded[..., 0]  # [B, E_gkt]
        gkt_target_indices = gkt_indices_padded[..., 1]
        # print('dfa_processor line 84')  # checked
        # print(
        #     f'cfg_r_indices: {cfg_row_indices.shape}; cfg_c_indices: {cfg_col_indices.shape}; type: {cfg_col_indices.dtype}')
        # print(
        #     f'gkt_r_indices: {gkt_row_indices.shape}; gkt_c_indices: {gkt_col_indices.shape}; type: {gkt_col_indices.dtype}')

        cfg_z = jnp.concatenate([node_fts, hidden], axis=-1)  # [B, N, 2*hidden_dim]
        # print(f'dfa_processor line 91, cfg_z: {cfg_z.shape}')   # checked
        m = hk.Linear(self.out_size)
        skip = hk.Linear(self.out_size)

        a_1 = hk.Linear(self.nb_heads)
        a_2 = hk.Linear(self.nb_heads)
        a_e = hk.Linear(self.nb_heads)
        cfg_att_1 = a_1(cfg_z)  # [B, N, nb_heads]
        cfg_att_2 = a_2(cfg_z)  # [B, N, nb_heads]
        # print(f'cfg_processor line 100, cfg_att_1: {cfg_att_1.shape}; cfg_att_2: {cfg_att_2.shape}')    # checked
        cfg_logits = (jnp.take_along_axis(arr=cfg_att_1,
                                          indices=dfa_utils.dim_expand_to(cfg_source_indices,
                                                                          cfg_att_1),
                                          axis=1)  # [B, E_cfg, nb_heads]
                      +
                      jnp.take_along_axis(arr=cfg_att_2,
                                          indices=dfa_utils.dim_expand_to(cfg_target_indices,
                                                                          cfg_att_2),
                                          axis=1))  # [B, E_cfg, nb_heads]

        # print(f'dfa_processor line 114, shape of cfg_logits is: {cfg_logits.shape}')    # checked

        @jax.vmap
        def _cfg_unsorted_segment_softmax_batched(logits, segment_ids):
            return unsorted_segment_softmax(logits=logits,
                                            segment_ids=segment_ids,
                                            num_segments=nb_cfg_edges_padded)  # [B, E_cfg, nb_heads]

        cfg_coefs = _cfg_unsorted_segment_softmax_batched(logits=jax.nn.leaky_relu(cfg_logits),
                                                          segment_ids=cfg_source_indices)
        # [B, E_cfg, nb_heads]
        cfg_values = m(cfg_z)  # [B, N, hidden_dim]
        # print(f'dfa_processor line 130, cfg_values: {cfg_values.shape}; cfg_coefs: {cfg_coefs.shape}')  # checked
        cfg_values = jnp.reshape(
            cfg_values,
            cfg_values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, nb_heads, hidden_dim/nb_heads]
        # print(f'dfa_processor line 134, cfg_values: {cfg_values.shape}')    # checked
        cfg_values_source = jnp.take_along_axis(arr=cfg_values,
                                                indices=dfa_utils.dim_expand_to(cfg_source_indices, cfg_values),
                                                axis=1)  # [B, E_cfg, nb_heads, hidden_dim/nb_heads]
        # print(f'dfa_processor line 139, cfg_values_source: {cfg_values_source.shape}; cfg_coefs: {cfg_coefs.shape}')    # checked
        cfg_hidden = jnp.expand_dims(cfg_coefs,
                                     axis=-1) * cfg_values_source  # [B, E_cfg, nb_heads, hidden_dim/nb_heads]

        # print(f'dfa_processor line 141, shape of cfg_hidden is: {cfg_hidden.shape}')    # checked

        @jax.vmap
        def _segment_sum_batched(data,  # [E, nb_heads, hidden_dim/nb_heads]
                                 segment_ids  # [E, ]
                                 ):
            # print(f'dfa_processor line 140, \ndata: {data.shape}; \nsegment_ids: {segment_ids.shape}')    # checked
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes_padded)

        cfg_hidden = _segment_sum_batched(data=cfg_hidden,
                                          segment_ids=cfg_target_indices)  # [B, N, nb_heads, hidden_dim/nb_heads]
        # print(f'dfa_processor line 155, shape of cfg_hidden is: {cfg_hidden.shape}')    # checked
        cfg_hidden = jnp.reshape(cfg_hidden, cfg_hidden.shape[:-2] + (self.out_size,))  # [B, N, hidden_dim]
        # print(f'dfa_processor line 145, shape of cfg_hidden is: {cfg_hidden.shape}')    # checked

        if self.residual:
            cfg_hidden += skip(cfg_z)

        if self.activation is not None:
            cfg_hidden = self.activation(cfg_hidden)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            cfg_hidden = ln(cfg_hidden)
        #     TODO(YZD)以上三个if是不是不应该出现在这里？

        gkt_z = jnp.concatenate([node_fts, cfg_hidden], axis=-1)  # [B, N, 2*hidden_dim]
        gkt_att_1 = a_1(gkt_z)  # [B, N, nb_heads]
        gkt_att_2 = a_2(gkt_z)  # [B, N, nb_heads]
        # TODO(YZD)这里应该和cfg_z共享a_1和a_2吗？
        gkt_att_e = a_e(gkt_edge_fts)  # [B, E_gkt, nb_heads]
        gkt_logits = (
                jnp.take_along_axis(arr=gkt_att_1,
                                    indices=dfa_utils.dim_expand_to(gkt_source_indices, gkt_att_1),
                                    axis=1)
                +  # + [B, E_gkt, nb_heads]
                jnp.take_along_axis(arr=gkt_att_2,
                                    indices=dfa_utils.dim_expand_to(gkt_target_indices, gkt_att_2),
                                    axis=1)
                +  # + [B, E_gkt, nb_heads]
                gkt_att_e  # + [B, E_gkt, nb_heads]
        )  # = [B, E_gkt, nb_heads]

        @jax.vmap
        def _gkt_unsorted_segment_softmax_batched(logits, segment_ids):
            return unsorted_segment_softmax(logits=logits,
                                            segment_ids=segment_ids,
                                            num_segments=nb_gkt_edges_padded)

        gkt_coefs = _gkt_unsorted_segment_softmax_batched(logits=jax.nn.leaky_relu(gkt_logits),
                                                          segment_ids=gkt_source_indices)  # [B, E_gkt, nb_heads]

        gkt_values = m(gkt_z)  # [B, N, hidden_dim]
        # TODO(YZD)这里也跟前面的cfg_z共享了m，不确定是否合理
        gkt_values = jnp.reshape(
            gkt_values,
            gkt_values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, nb_heads, hidden_dim/nb_heads]
        gkt_values_source = jnp.take_along_axis(arr=gkt_values,
                                                indices=dfa_utils.dim_expand_to(gkt_source_indices, gkt_values),
                                                axis=1)  # [B, E_gkt, nb_heads, hidden_dim/nb_heads]

        ret = jnp.expand_dims(gkt_coefs, axis=-1) * gkt_values_source
        # [B, E_gkt, nb_heads, 1] * [B, E_gkt, nb_heads, hidden_dim/nb_heads] = [B, E_gkt, nb_heads, hidden_dim/nb_heads]
        ret = _segment_sum_batched(data=ret,
                                   segment_ids=gkt_target_indices)  # [B, N, nb_heads, hidden_dim/nb_heads]
        ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [B, N, hidden_dim]

        if self.residual:
            ret += skip(gkt_z)

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)
        # print(f'dfa_processor line 221, ret: {ret.shape}')  # [B, N, hidden_dim] [2, 150, 16]   # checked
        return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


def unsorted_segment_softmax(logits,  # [E, nb_heads]
                             segment_ids,  # [E, ]
                             num_segments):  # E
    """Returns softmax over each segment.

    Args:
      logits: Logits of elements, of shape `[dim_0, ...]`.
      segment_ids: Segmentation of logits, which elements will participate in
        the same softmax. Shape `[dim_0]`.
      num_segments: Scalar number of segments, typically `max(segment_ids)`.

    Returns:
      Probabilities of the softmax, shape `[dim_0, ...]`.
    """
    # print(
    #     f'dfa_processor line 231, \nlogits: {logits.shape}; \nsegment_ids: {segment_ids.shape}; \nnum_segments = {num_segments}')   # checked
    segment_max_ = jax.ops.segment_max(logits, segment_ids, num_segments)
    broadcast_segment_max = segment_max_[segment_ids]
    shifted_logits = logits - broadcast_segment_max

    # Sum and get the probabilities.
    exp_logits = jnp.exp(shifted_logits)
    exp_sum = jax.ops.segment_sum(exp_logits, segment_ids, num_segments)
    broadcast_exp_sum = exp_sum[segment_ids]
    probs = exp_logits / broadcast_exp_sum
    return probs

class AlignGNN(DFAProcessor):
    def __init__(self,
                 out_size: int,
                 name: str = 'gnn_align',
                 ):
        super().__init__(name=name)
        self.out_size = out_size

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            hidden: _chex_Array,  # [B, N， m, hidden_dim]
            cfg_indices_padded: _chex_Array,  # [B, E, 2]
            node_fts: _chex_Array,  # [B, N， m, hidden_dim]
            edge_fts: _chex_Array,  # [B, E, hidden_dim]    cfg_edge
    ):
        nb_nodes = node_fts.shape[1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, E]
        edge_indices_target = cfg_indices_padded[..., 1]

        # node_fts||hidden -> nh_fts
        node_hidden_fts = jnp.concatenate([node_fts, hidden], axis=-1)  # [B, N, m, 2*hidden_dim]
        # [B, N， m, 2*hidden_dim]
        fuse_nh_linear = hk.Linear(self.out_size)
        # inject gen/kill/trace_i info into hidden
        nh_fts = fuse_nh_linear(node_hidden_fts)
        # [B, N, m, out_size]
        nh_values = jnp.take_along_axis(arr=nh_fts,  # [B, N, m, out_size]
                                        indices=dfa_utils.dim_expand_to(edge_indices_source, nh_fts),
                                        # [B, E, 1, 1]
                                        axis=1)
        # [B, E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, E, hidden_dim] -> [B, E, 1]
        filtered_nh_values = jnp.expand_dims(edge_coeff, axis=-1) * nh_values

        # [B, E, 1, 1] * [B, E, m, hidden_dim] -> [B, E, m, hidden_dim]

        @jax.vmap
        def _segment_max_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        updated_bits = _segment_max_batched(data=filtered_nh_values,  # [B, E, m, hidden_dim]
                                            segment_ids=edge_indices_target)
        # [B, N, m, hidden_dim]
        return updated_bits

class GNNV3(DFAProcessor):
    def __init__(self, name: str = 'gnn_v3'):
        super().__init__(name=name)

    def __call__(self,
                 cfg_indices_padded: _chex_Array,  # [B, E, 2]
                 trace_h_i,  # [B, N, m]
                 gen_vectors,  # [B, N, m]
                 kill_vectors,  # [B, N, m]
                 direction,  # [B, 2E]
                 cfg_edges,  # [B, 2E]
                 ):
        nb_nodes = trace_h_i.shape[1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        trace_h_i_sources = jnp.take_along_axis(arr=trace_h_i,  # [B, N, m]
                                                indices=dfa_utils.dim_expand_to(edge_indices_source, trace_h_i),
                                                # [B, 2E, 1]
                                                axis=1)
        # [B, 2E, m]
        # print(f'dfa_processer line 61, trace_h_i: {trace_h_i_sources.shape}')
        # direction || cfg_edges -> coef
        concated_de = jnp.concatenate([jnp.expand_dims(direction, axis=-1), jnp.expand_dims(cfg_edges, axis=-1)],
                                      axis=-1)  # [B, 2E, 1] || [B, 2E, 1] -> [B, 2E, 2]
        edge_coeff_linear = hk.Linear(1)
        edge_coeff = edge_coeff_linear(concated_de)  # [B, 2E, 1]
        filtered_trace_h_i = edge_coeff * trace_h_i_sources  # [B, 2E, m]

        @jax.vmap
        def _segment_max_batched(data,  # [2E, m]
                                 segment_ids  # [2E, ]
                                 ):
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        meeted_state = _segment_max_batched(data=filtered_trace_h_i,  # [B, 2E, m]
                                            segment_ids=edge_indices_target)
        # [B, N, m]
        concated_gen_kill_state = jnp.concatenate([jnp.expand_dims(gen_vectors, axis=-1),
                                                   jnp.expand_dims(kill_vectors, axis=-1),
                                                   jnp.expand_dims(meeted_state, axis=-1)],
                                                  axis=-1)
        # [B, N, m, 3]
        update_func = hk.Linear(1)
        return update_func(concated_gen_kill_state)  # [B, N, m, 1]

class GNNV4_sum(DFAProcessor):
    def __init__(self, name: str = 'gnn_v4'):
        super(GNNV4_sum, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hint_state: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        hint_sources = jnp.take_along_axis(arr=hint_state,  # [B, N, m, hidden_dim]
                                           indices=dfa_utils.dim_expand_to(edge_indices_source, hint_state),
                                           # [B, 2E, 1, 1]
                                           axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        hint_sources = jnp.expand_dims(edge_coeff, axis=-1) * hint_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_max_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_hint = _segment_max_batched(data=hint_sources,
                                               segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        nh_concated = jnp.concatenate([node_fts, aggregated_hint], axis=-1)
        update_linear = hk.Linear(hidden_dims)  # w: [2*hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(nh_concated)
        return updated_hidden

class GNNV4_max(DFAProcessor):
    def __init__(self, name: str = 'gnn_v4'):
        super(GNNV4_max, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hint_state: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        hint_sources = jnp.take_along_axis(arr=hint_state,  # [B, N, m, hidden_dim]
                                           indices=dfa_utils.dim_expand_to(edge_indices_source, hint_state),
                                           # [B, 2E, 1, 1]
                                           axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        hint_sources = jnp.expand_dims(edge_coeff, axis=-1) * hint_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_max_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_max(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_hint = _segment_max_batched(data=hint_sources,
                                               segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        nh_concated = jnp.concatenate([node_fts, aggregated_hint], axis=-1)
        update_linear = hk.Linear(hidden_dims)  # w: [2*hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(nh_concated)
        return updated_hidden

class GNNV4_min(DFAProcessor):
    def __init__(self, name: str = 'gnn_v4'):
        super(GNNV4_min, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hint_state: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        hint_sources = jnp.take_along_axis(arr=hint_state,  # [B, N, m, hidden_dim]
                                           indices=dfa_utils.dim_expand_to(edge_indices_source, hint_state),
                                           # [B, 2E, 1, 1]
                                           axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        hint_sources = jnp.expand_dims(edge_coeff, axis=-1) * hint_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_min_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_min(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_hint = _segment_min_batched(data=hint_sources,
                                               segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        nh_concated = jnp.concatenate([node_fts, aggregated_hint], axis=-1)
        update_linear = hk.Linear(hidden_dims)  # w: [2*hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(nh_concated)
        return updated_hidden

class GNNV6_sum(DFAProcessor):
    def __init__(self, name: str = 'gnn_v6_sum'):
        super(GNNV6_sum, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hint_state: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        hint_sources = jnp.take_along_axis(arr=hint_state,  # [B, N, m, hidden_dim]
                                           indices=dfa_utils.dim_expand_to(edge_indices_source, hint_state),
                                           # [B, 2E, 1, 1]
                                           axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        hint_sources = jnp.expand_dims(edge_coeff, axis=-1) * hint_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_max_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_hint = _segment_max_batched(data=hint_sources,
                                               segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        update_linear = hk.Linear(hidden_dims)  # w: [hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(node_fts + aggregated_hint)
        return updated_hidden

class GNNV6_max(DFAProcessor):
    def __init__(self, name: str = 'gnn_v6_max'):
        super(GNNV6_max, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hint_state: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        hint_sources = jnp.take_along_axis(arr=hint_state,  # [B, N, m, hidden_dim]
                                           indices=dfa_utils.dim_expand_to(edge_indices_source, hint_state),
                                           # [B, 2E, 1, 1]
                                           axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        hint_sources = jnp.expand_dims(edge_coeff, axis=-1) * hint_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_max_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_max(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_hint = _segment_max_batched(data=hint_sources,
                                               segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        update_linear = hk.Linear(hidden_dims)  # w: [hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(node_fts + aggregated_hint)
        return updated_hidden

class GNNV6_min(DFAProcessor):
    def __init__(self, name: str = 'gnn_v6'):
        super(GNNV6_min, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hint_state: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        hint_sources = jnp.take_along_axis(arr=hint_state,  # [B, N, m, hidden_dim]
                                           indices=dfa_utils.dim_expand_to(edge_indices_source, hint_state),
                                           # [B, 2E, 1, 1]
                                           axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        hint_sources = jnp.expand_dims(edge_coeff, axis=-1) * hint_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_min_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_min(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_hint = _segment_min_batched(data=hint_sources,
                                               segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        update_linear = hk.Linear(hidden_dims)  # w: [2*hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(node_fts + aggregated_hint)
        return updated_hidden


DFAProcessorFactory = Callable[[int], DFAProcessor]

def get_dfa_processor_factory(kind: str,
                              aggregator: str = 'sum',
                              *args,
                              **kwargs,
                              ) -> DFAProcessorFactory:
    """Returns a processor factory.

    Args:
      kind: One of the available types of processor.
    Returns:
      A callable that takes an `out_size` parameter (equal to the hidden
      dimension of the network) and returns a processor instance.
    """

    def _dfa_factory(out_size: int):
        if kind == 'gnn_v1':
            processor = GATSparse(out_size=out_size,
                                  *args,
                                  **kwargs)
        elif kind == 'gnn_v2':
            processor = AlignGNN(out_size=out_size)
        elif kind == 'gnn_v3':
            processor = GNNV3()
        elif kind == 'gnn_v4' or kind == 'gnn_v5':
            if aggregator == 'sum':
                processor = GNNV4_sum()
            elif aggregator == 'max':
                processor = GNNV4_max()
            elif aggregator == 'min':
                processor = GNNV4_min()
            else:
                raise ValueError('Unexpected aggregator: ' + aggregator)
        elif kind == 'gnn_v6' or kind == 'gnn_v7':
            if aggregator == 'sum':
                processor = GNNV6_sum()
            elif aggregator == 'max':
                processor = GNNV6_max()
            elif aggregator == 'min':
                processor = GNNV6_min()
            else:
                raise ValueError('Unexpected aggregator: ' + aggregator)
        else:
            raise ValueError('Unexpected processor kind ' + kind)

        return processor

    return _dfa_factory
