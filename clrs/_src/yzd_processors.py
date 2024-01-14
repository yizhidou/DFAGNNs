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
    def __call__(
            self,
            node_fts: _chex_Array,
            hidden: _chex_Array,
            cfg_indices_padded: _chex_Array,
            edge_fts: _chex_Array
    ) -> Tuple[_chex_Array, Optional[_chex_Array]]:
        """Processor inference step.

        Returns:
          Output of processor inference step as a 2-tuple of (node, edge)
          embeddings. The edge embeddings can be None.
        """
        pass


class AlignGNN_v1(DFAProcessor):
    def __init__(self,
                 nb_bits_each_node: int,
                 activation: Optional[_Fn] = jax.nn.relu,
                 name: str = 'gat_sparse',
                 ):
        super().__init__(name=name)
        self.nb_bits_each = nb_bits_each_node
        self.activation = activation

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            bit_fts: _chex_Array,  # [B, N*m, hidden_dim]
            hidden: _chex_Array,  # [B, N*m, hidden_dim]
            edge_indices: _chex_Array,  # [B, E, 2]
            edge_fts: _chex_Array,  # [B, E, hidden_dim]    cfg_edge
    ):
        batch_size, nb_bits_total, hidden_dim = bit_fts.shape
        nb_edges = edge_indices.shape[1]
        assert edge_fts.shape[:-1] == (batch_size, nb_edges, hidden_dim)

        # derive bit_indices from edge_indices
        bit_indices_base = jnp.arange(self.nb_bits_each).reshape(1, -1)  # [1, m]
        edge_indices_source = edge_indices[..., 0]  # [B, E]
        bit_indices_source = jnp.expand_dims(edge_indices_source * self.nb_bits_each, axis=-1) + jnp.expand_dims(
            bit_indices_base, axis=0)
        # [B, E, 1] + [1, 1, m] -> [B, E, m]
        bit_indices_source = bit_indices_source.reshape((batch_size, -1))  # [B, E*m]
        edge_indices_target = edge_indices[..., 1]
        bit_indices_target = jnp.expand_dims(edge_indices_target * self.nb_bits_each, axis=-1) + jnp.expand_dims(
            bit_indices_base,
            axis=0)
        # [B, E, 1] + [1, 1, m] -> [B, E, m]
        bit_indices_target = bit_indices_target.reshape((batch_size, -1))

        # bit_fts||hidden -> bh_fts
        bit_hidden_fts = jnp.concatenate([bit_fts, hidden], axis=-1)  # [B, N*m, 2*hidden_dim]
        fuse_bit_hidden_linear = hk.Linear(hidden_dim)
        bh_fts = fuse_bit_hidden_linear(bit_hidden_fts)
        # [B, N*m, hidden_dim]
        bh_values = jnp.take_along_axis(arr=bh_fts,
                                        indices=dfa_utils.dim_expand_to(bit_indices_source, bit_hidden_fts),
                                        axis=1)  # [B, E*m, hidden_dim]

        # get coefficient from edge_fts: [B, E, hidden_dim] -> [B, E*m]
        edge_coeff_linear = hk.Linear(1)
        edge_coeff = edge_coeff_linear(edge_fts)
        # [B, E, 1]
        bit_coeff = jnp.reshape(a=edge_coeff, repeats=self.nb_bits_each, axis=1)

        # [B, E*m, 1]

        @jax.vmap
        def _segment_max_batched(data,  # [E*m, hidden_dim]
                                 segment_ids  # [E*m, ]
                                 ):
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_bits_total)

        updated_bits = _segment_max_batched(data=bit_coeff * bh_values,  # [B, E*m, hidden_dim]
                                            segment_ids=bit_indices_target)
        # [B, N*m, hidden_dim]
        return updated_bits


class AlignGNN_v2(DFAProcessor):
    def __init__(self,
                 out_size: int,
                 # activation: Optional[_Fn] = jax.nn.relu,
                 name: str = 'gat_sparse',
                 ):
        super().__init__(name=name)
        self.out_size = out_size
        # self.activation = activation

    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            node_fts: _chex_Array,  # [B, N， m, hidden_dim]
            hidden: _chex_Array,  # [B, N， m, hidden_dim]
            edge_indices: _chex_Array,  # [B, E, 2]
            edge_fts: _chex_Array,  # [B, E, hidden_dim]    cfg_edge
    ):
        nb_nodes = node_fts.shape[1]
        edge_indices_source = edge_indices[..., 0]  # [B, E]
        edge_indices_target = edge_indices[..., 1]

        # node_fts||hidden -> nh_fts
        node_hidden_fts = jnp.concatenate([node_fts, hidden], axis=-1)  # [B, N*m, 2*hidden_dim]
        # [B, N， m, 2*hidden_dim]
        fuse_nh_linear = hk.Linear(self.out_size)
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


DFAProcessorFactory = Callable[[int], DFAProcessor]


def get_dfa_processor_factory(kind: str,
                              nb_heads: int,
                              activation: Optional[_Fn],
                              residual: bool,
                              use_ln: bool) -> DFAProcessorFactory:
    """Returns a processor factory.

    Args:
      kind: One of the available types of processor.
      use_ln: Whether the processor passes the output through a layernorm layer.
      nb_triplet_fts: How many triplet features to compute.
      nb_heads: Number of attention heads for GAT processors.
    Returns:
      A callable that takes an `out_size` parameter (equal to the hidden
      dimension of the network) and returns a processor instance.
    """

    def _dfa_factory(out_size: int):
        if kind == 'dfa_gat':
            processor = AlignGNN_v2(out_size=out_size)
        else:
            raise ValueError('Unexpected processor kind ' + kind)

        return processor

    return _dfa_factory
