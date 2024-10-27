import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.ops

import abc
import math
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

class DFAGNN_plus_max(DFAProcessor):
    def __init__(self, name: str = 'dfa_gnn_plus_max'):
        super(GNNV6_max, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hint_state: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        hint_sources = jnp.take_along_axis(arr=hint_state,  # [B, N, m, hidden_dim]
                                           indices=dfa_utils.dim_expand_to(edge_indices_source, hint_state),
                                           # [B, 2E, 1, 1]
                                           axis=1)

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

class DFAGNN_plus_sum(DFAProcessor):
    def __init__(self, name: str = 'dfa_gnn_plus_sum'):
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
        def _segment_sum_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_hint = _segment_sum_batched(data=hint_sources,
                                               segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        update_linear = hk.Linear(hidden_dims)  # w: [hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(node_fts + aggregated_hint)
        return updated_hidden

class DFAGNN_plus_mean(DFAProcessor):
    def __init__(self, name: str = 'dfa_gnn_plus_mean'):
        super(GNNV6_mean, self).__init__(name=name)

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
        def _segment_mean_batched(data,  # [E, m, hidden_dim]
                                  segment_ids  # [E, ]
                                  ):
            segment_counts = jax.ops.segment_sum(data=jnp.ones_like(data),
                                                 segment_ids=segment_ids,
                                                 num_segments=nb_nodes)
            segment_sums = jax.ops.segment_sum(data=data,
                                               segment_ids=segment_ids,
                                               num_segments=nb_nodes)
            return segment_sums / segment_counts
        aggregated_hint = _segment_mean_batched(data=hint_sources,
                                               segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        update_linear = hk.Linear(hidden_dims)  # w: [hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(node_fts + aggregated_hint)
        return updated_hidden

class DFAGNN_max(DFAProcessor):
    def __init__(self, name: str = 'dfa_gnn_max'):
        super(GNNV8_max, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hidden: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        nh_concated = jnp.concatenate([node_fts, hidden], axis=-1)
        nh_transform_linear = hk.Linear(hidden_dims)
        nh_fused = nh_transform_linear(nh_concated)

        nh_sources = jnp.take_along_axis(arr=nh_fused,  # [B, N, m, hidden_dim]
                                         indices=dfa_utils.dim_expand_to(edge_indices_source, nh_fused),
                                         # [B, 2E, 1, 1]
                                         axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        nh_sources = jnp.expand_dims(edge_coeff, axis=-1) * nh_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_max_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_max(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_nh = _segment_max_batched(data=nh_sources,
                                             segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        update_linear = hk.Linear(hidden_dims)  # w: [hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(aggregated_nh)
        return updated_hidden

class DFAGNN_sum(DFAProcessor):
    def __init__(self, name: str = 'dfa_gnn_sum'):
        super(GNNV8_sum, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hidden: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        nh_concated = jnp.concatenate([node_fts, hidden], axis=-1)
        nh_transform_linear = hk.Linear(hidden_dims)
        nh_fused = nh_transform_linear(nh_concated)

        nh_sources = jnp.take_along_axis(arr=nh_fused,  # [B, N, m, hidden_dim]
                                         indices=dfa_utils.dim_expand_to(edge_indices_source, nh_fused),
                                         # [B, 2E, 1, 1]
                                         axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        nh_sources = jnp.expand_dims(edge_coeff, axis=-1) * nh_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_sum_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_nh = _segment_sum_batched(data=nh_sources,
                                             segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        update_linear = hk.Linear(hidden_dims)  # w: [hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(aggregated_nh)
        return updated_hidden

class DFAGNN_mean(DFAProcessor):
    def __init__(self, name: str = 'dfa_gnn_mean'):
        super(GNNV8_mean, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hidden: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        nb_nodes, hidden_dims = node_fts.shape[1], node_fts.shape[-1]
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        nh_concated = jnp.concatenate([node_fts, hidden], axis=-1)
        nh_transform_linear = hk.Linear(hidden_dims)
        nh_fused = nh_transform_linear(nh_concated)

        nh_sources = jnp.take_along_axis(arr=nh_fused,  # [B, N, m, hidden_dim]
                                         indices=dfa_utils.dim_expand_to(edge_indices_source, nh_fused),
                                         # [B, 2E, 1, 1]
                                         axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        nh_sources = jnp.expand_dims(edge_coeff, axis=-1) * nh_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_mean_batched(data,  # [E, m, hidden_dim]
                                  segment_ids  # [E, ]
                                  ):
            segment_counts = jax.ops.segment_sum(data=jnp.ones_like(data),
                                                 segment_ids=segment_ids,
                                                 num_segments=nb_nodes)
            segment_sums = jax.ops.segment_sum(data=data,
                                               segment_ids=segment_ids,
                                               num_segments=nb_nodes)
            return segment_sums / segment_counts

        aggregated_nh = _segment_mean_batched(data=nh_sources,
                                              segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]
        update_linear = hk.Linear(hidden_dims)  # w: [hidden_dim, hidden_dim]; b: [hidden_dim]
        updated_hidden = update_linear(aggregated_nh)
        return updated_hidden

class DFAGNN_minus_max(DFAProcessor):
    def __init__(self, name: str = 'dfa_gnn_minus_max'):
        super(GNNV11_max, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hidden: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        _, nb_nodes, nb_ip, hidden_dims = node_fts.shape
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        nh_concated = jnp.concatenate([node_fts, hidden], axis=-1)
        # [B, N, m, 2 * hidden_dim]
        nh_transform_linear = hk.Linear(hidden_dims)
        nh_fused = nh_transform_linear(nh_concated)
        # [B, N, m, hidden_dim]

        nh_sources = jnp.take_along_axis(arr=nh_fused,  # [B, N, m, hidden_dim]
                                         indices=dfa_utils.dim_expand_to(edge_indices_source, nh_fused),
                                         # [B, 2E, 1, 1]
                                         axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        nh_sources = jnp.expand_dims(edge_coeff, axis=-1) * nh_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_max_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_max(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_nh = _segment_max_batched(data=nh_sources,
                                             segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]

        # cancel the independence requirement in update
        Linear_Q_update = hk.Linear(hidden_dims) 
        Linear_K_update = hk.Linear(hidden_dims)
        Linear_V_update = hk.Linear(hidden_dims)
        Q_update = Linear_Q_update(aggregated_nh)  # [B, N, m, hidden_dim]
        K_update = Linear_K_update(aggregated_nh)
        V_update = Linear_V_update(aggregated_nh)
        QK_update = jnp.matmul(Q_update, jnp.transpose(K_update, (0 ,1, 3, 2))) / math.sqrt(hidden_dims)
        # [B, N, m, m]
        softmax_QK_update = jax.nn.softmax(QK_update)
        # [B, N, m, m]
        # aggregated_nh = jnp.matmul(softmax_QK_update, V_update)
        # # [B, N, m, h]
        updated_hidden = jnp.matmul(softmax_QK_update, V_update)
        # [B, N, m, h]

        return updated_hidden

class DFAGNN_minus_sum(DFAProcessor):
    def __init__(self, name: str = 'dfa_gnn_minus_sum'):
        super(GNNV11_sum, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hidden: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        _, nb_nodes, nb_ip, hidden_dims = node_fts.shape
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        nh_concated = jnp.concatenate([node_fts, hidden], axis=-1)
        # [B, N, m, 2 * hidden_dim]
        nh_transform_linear = hk.Linear(hidden_dims)
        nh_fused = nh_transform_linear(nh_concated)
        # [B, N, m, hidden_dim]

        nh_sources = jnp.take_along_axis(arr=nh_fused,  # [B, N, m, hidden_dim]
                                         indices=dfa_utils.dim_expand_to(edge_indices_source, nh_fused),
                                         # [B, 2E, 1, 1]
                                         axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        nh_sources = jnp.expand_dims(edge_coeff, axis=-1) * nh_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_sum_batched(data,  # [E, m, hidden_dim]
                                 segment_ids  # [E, ]
                                 ):
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes)

        aggregated_nh = _segment_sum_batched(data=nh_sources,
                                             segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]

        # cancel the independence requirement in update
        Linear_Q_update = hk.Linear(hidden_dims)
        Linear_K_update = hk.Linear(hidden_dims)
        Linear_V_update = hk.Linear(hidden_dims)
        Q_update = Linear_Q_update(aggregated_nh)  # [B, N, m, hidden_dim]
        K_update = Linear_K_update(aggregated_nh)
        V_update = Linear_V_update(aggregated_nh)
        QK_update = jnp.matmul(Q_update, jnp.transpose(K_update, (0, 1, 3, 2))) / math.sqrt(hidden_dims)
        # [B, N, m, m]
        softmax_QK_update = jax.nn.softmax(QK_update)
        # [B, N, m, m]
        # aggregated_nh = jnp.matmul(softmax_QK_update, V_update)
        # # [B, N, m, h]
        updated_hidden = jnp.matmul(softmax_QK_update, V_update)
        # [B, N, m, h]

        return updated_hidden

class DFAGNN_minus_mean(DFAProcessor):
    def __init__(self, name: str = 'dfa_gnn_minus_mean'):
        super(GNNV11_mean, self).__init__(name=name)

    def __call__(self, cfg_indices_padded: _chex_Array,  # [B, E, 2],
                 hidden: _chex_Array,  # [B, N, m, hidden_dim]
                 node_fts: _chex_Array,  # [B, N， m, hidden_dim]
                 edge_fts: _chex_Array,  # [B, E, hidden_dim]
                 ):
        # print(f'dfa_processor line 49, hint_state: {hint_state.shape}')
        _, nb_nodes, nb_ip, hidden_dims = node_fts.shape
        edge_indices_source = cfg_indices_padded[..., 0]  # [B, 2E]
        edge_indices_target = cfg_indices_padded[..., 1]

        nh_concated = jnp.concatenate([node_fts, hidden], axis=-1)
        # [B, N, m, 2 * hidden_dim]
        nh_transform_linear = hk.Linear(hidden_dims)
        nh_fused = nh_transform_linear(nh_concated)
        # [B, N, m, hidden_dim]

        nh_sources = jnp.take_along_axis(arr=nh_fused,  # [B, N, m, hidden_dim]
                                         indices=dfa_utils.dim_expand_to(edge_indices_source, nh_fused),
                                         # [B, 2E, 1, 1]
                                         axis=1)
        #   [B, 2E, m, hidden_dim]

        # get coefficient from edge_fts
        edge_coeff_linear = hk.Linear(1)  # w: [hidden_dim, 1]; b: [1, ]
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, 2E, hidden_dim] -> [B, 2E, 1]

        nh_sources = jnp.expand_dims(edge_coeff, axis=-1) * nh_sources

        # [B, 2E, 1, 1] * [B, 2E, m, hidden_dim] -> [B, 2E, m, hidden_dim]

        @jax.vmap
        def _segment_mean_batched(data,  # [E, m, hidden_dim]
                                  segment_ids  # [E, ]
                                  ):
            segment_counts = jax.ops.segment_sum(data=jnp.ones_like(data),
                                                 segment_ids=segment_ids,
                                                 num_segments=nb_nodes)
            segment_sums = jax.ops.segment_sum(data=data,
                                               segment_ids=segment_ids,
                                               num_segments=nb_nodes)
            return segment_sums / segment_counts

        aggregated_nh = _segment_mean_batched(data=nh_sources,
                                             segment_ids=edge_indices_target)
        #   [B, N, m, hidden_dim]

        # cancel the independence requirement in update
        Linear_Q_update = hk.Linear(hidden_dims)
        Linear_K_update = hk.Linear(hidden_dims)
        Linear_V_update = hk.Linear(hidden_dims)
        Q_update = Linear_Q_update(aggregated_nh)  # [B, N, m, hidden_dim]
        K_update = Linear_K_update(aggregated_nh)
        V_update = Linear_V_update(aggregated_nh)
        QK_update = jnp.matmul(Q_update, jnp.transpose(K_update, (0, 1, 3, 2))) / math.sqrt(hidden_dims)
        # [B, N, m, m]
        softmax_QK_update = jax.nn.softmax(QK_update)
        # [B, N, m, m]
        # aggregated_nh = jnp.matmul(softmax_QK_update, V_update)
        # # [B, N, m, h]
        updated_hidden = jnp.matmul(softmax_QK_update, V_update)
        # [B, N, m, h]

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
        if kind == 'DFAGNN_plus':
            if aggregator == 'sum':
                processor = DFAGNN_plus_sum()
            elif aggregator == 'mean':
                processor = DFAGNN_plus_mean()
            elif aggregator == 'max':
                processor = DFAGNN_plus_max()
            else:
                raise ValueError('Unexpected aggregator: ' + aggregator)
        elif kind == 'DFAGNN':
            if aggregator == 'sum':
                processor = DFAGNN_sum()
            elif aggregator == 'mean':
                processor = DFAGNN_mean()
            elif aggregator == 'max':
                processor = DFAGNN_max()
            else:
                raise ValueError('Unexpected aggregator: ' + aggregator)
        elif kind == 'DFAGNN_minus':
            if aggregator == 'sum':
                processor = DFAGNN_minus_sum()
            elif aggregator == 'mean':
                processor = DFAGNN_minus_mean()
            elif aggregator == 'max':
                processor = DFAGNN_minus_max()
            else:
                raise ValueError('Unexpected aggregator: ' + aggregator)
        else:
            raise ValueError('Unexpected processor kind ' + kind)

        return processor

    return _dfa_factory

