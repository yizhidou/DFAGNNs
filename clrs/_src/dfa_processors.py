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
                 hidden: _chex_Array,
                 *args,
                 **kwargs):
        """Processor inference step.

        Returns:
          Output of processor inference step as a 2-tuple of (node, edge)
          embeddings. The edge embeddings can be None.
        """
        pass


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
        node_hidden_fts = jnp.concatenate([node_fts, hidden], axis=-1)  # [B, N*m, 2*hidden_dim]
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


DFAProcessorFactory = Callable[[int], DFAProcessor]


def get_dfa_processor_factory(kind: str,
                              *args,
                              **kwargs) -> DFAProcessorFactory:
    """Returns a processor factory.

    Args:
      kind: One of the available types of processor.
    Returns:
      A callable that takes an `out_size` parameter (equal to the hidden
      dimension of the network) and returns a processor instance.
    """

    def _dfa_factory(out_size: int):
        if kind == 'gnn_align':
            processor = AlignGNN(out_size=out_size)
        elif kind == 'dfa_gat':
            processor = GATSparse(out_size=out_size,
                                  *args,
                                  **kwargs)
        else:
            raise ValueError('Unexpected processor kind ' + kind)

        return processor

    return _dfa_factory
