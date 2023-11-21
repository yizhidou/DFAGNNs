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
            gkt_indices_padded: _chex_Array,
            gkt_edge_fts: Union[_chex_Array, None] = None,
            gen_dp_data: Union[_chex_Array, None] = None,
            kill_dp_data: Union[_chex_Array, None] = None,
            trace_h_i_dp_data: Union[_chex_Array, None] = None
    ) -> Tuple[_chex_Array, Optional[_chex_Array]]:
        """Processor inference step.

        Returns:
          Output of processor inference step as a 2-tuple of (node, edge)
          embeddings. The edge embeddings can be None.
        """
        pass


class YZDProcessor(DFAProcessor):
    def __init__(self, out_size: int, name='yzd_gnn'):
        super().__init__(name=name)
        self.out_size = out_size

    def __call__(self,
                 node_fts: _chex_Array,  # [B, N, node_fts_dim]
                 hidden: _chex_Array,  # [B, E, hidden_dim]
                 gen_dp_data: _chex_Array,  # [B, E_gkt]
                 kill_dp_data: _chex_Array,  # [B, E_gkt]
                 trace_h_i_dp_data: _chex_Array,  # [B, E_gkt]
                 cfg_indices_padded: _chex_Array,  # [B, E_cfg, 2]
                 gkt_indices_padded: _chex_Array,  # [B, E_gkt, 2]
                 ):
        nb_nodes_padded = node_fts.shape[-2]

        def _segment_sum_batched(data,  # [E, nb_heads, hidden_dim/nb_heads]
                                 segment_ids  # [E, ]
                                 ):
            # print(f'dfa_processor line 140, \ndata: {data.shape}; \nsegment_ids: {segment_ids.shape}')    # checked
            return jax.ops.segment_sum(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes_padded)

        def _segment_max_batched(data,
                                 segment_ids):
            return jax.ops.segment_max(data=data,
                                       segment_ids=segment_ids,
                                       num_segments=nb_nodes_padded)

        # step 1: info ip -> pp filtered by trace_h_i
        gkt_source_indices = gkt_indices_padded[..., 0]  # [B, E_gkt]
        gkt_target_indices = gkt_indices_padded[..., 1]
        source_msg_1 = jnp.take_along_axis(arr=node_fts,
                                           indices=dfa_utils.dim_expand_to(gkt_source_indices, node_fts),
                                           axis=1)  # [B, E_gkt, node_fts_dim]
        filtered_msg_1 = dfa_utils.dim_expand_to(trace_h_i_dp_data, source_msg_1) * source_msg_1  # [B, E_gkt, node_fts_dim]
        node_msg_1 = _segment_sum_batched(data=filtered_msg_1,
                                          segment_ids=gkt_target_indices)  # [B, N, node_fts_dim]
        hidden_1 = jnp.concatenate([node_msg_1, hidden], axis=-1)  # [B, N, node_fts_dim+hidden_dim]
        # step 2: info pp -> pp
        cfg_source_indices = cfg_indices_padded[..., 0]  # [B, E_gkt]
        cfg_target_indices = cfg_indices_padded[..., 1]
        source_msg_2 = jnp.take_along_axis(arr=hidden_1,
                                           indices=dfa_utils.dim_expand_to(cfg_source_indices, hidden_1),
                                           axis=1)  # [B, E_cfg, , node_fts_dim+hidden_dim]
        hidden_2 = _segment_max_batched(data=source_msg_2,
                                        segment_ids=cfg_target_indices)  # [B, N, node_fts_dim+hidden_dim]
        # step 3: info ip -> pp, filtered by kill
        # 这个我真的不确定是重新拿一份还是就用source_msg_1
        source_msg_3 = jnp.take_along_axis(arr=node_fts,
                                           indices=dfa_utils.dim_expand_to(gkt_source_indices, node_fts),
                                           axis=1)  # [B, E_gkt, node_fts_dim]
        filtered_msg_3 = dfa_utils.dim_expand_to(kill_dp_data,
                                            source_msg_3) * source_msg_3  # [B, E_gkt, , node_fts_dim]


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
                                                          segment_ids=cfg_row_indices)
        cfg_values = m(cfg_z)  # [B, N, hidden_dim]
        # print(f'dfa_processor line 130, cfg_values: {cfg_values.shape}; cfg_coefs: {cfg_coefs.shape}')  # checked
        cfg_values = jnp.reshape(
            cfg_values,
            cfg_values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, nb_heads, hidden_dim/nb_heads]
        # print(f'dfa_processor line 134, cfg_values: {cfg_values.shape}')    # checked
        cfg_values_source = jnp.take_along_axis(arr=cfg_values,
                                                indices=dfa_utils.dim_expand_to(cfg_col_indices, cfg_values),
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
                                          segment_ids=cfg_row_indices)  # [B, N, nb_heads, hidden_dim/nb_heads]
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
                                    indices=dfa_utils.dim_expand_to(gkt_row_indices, gkt_att_1),
                                    axis=1)
                +  # + [B, E_gkt, nb_heads]
                jnp.take_along_axis(arr=gkt_att_2,
                                    indices=dfa_utils.dim_expand_to(gkt_col_indices, gkt_att_2),
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
                                                          segment_ids=gkt_row_indices)  # [B, E_gkt, nb_heads]

        gkt_values = m(gkt_z)  # [B, N, hidden_dim]
        # TODO(YZD)这里也跟前面的cfg_z共享了m，不确定是否合理
        gkt_values = jnp.reshape(
            gkt_values,
            gkt_values.shape[:-1] + (self.nb_heads, self.head_size))  # [B, N, nb_heads, hidden_dim/nb_heads]
        gkt_values_source = jnp.take_along_axis(arr=gkt_values,
                                                indices=dfa_utils.dim_expand_to(gkt_col_indices, gkt_values),
                                                axis=1)  # [B, E_gkt, nb_heads, hidden_dim/nb_heads]

        ret = jnp.expand_dims(gkt_coefs, axis=-1) * gkt_values_source
        # [B, E_gkt, nb_heads, 1] * [B, E_gkt, nb_heads, hidden_dim/nb_heads] = [B, E_gkt, nb_heads, hidden_dim/nb_heads]
        ret = _segment_sum_batched(data=ret,
                                   segment_ids=gkt_row_indices)  # [B, N, nb_heads, hidden_dim/nb_heads]
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


# class DAFPGN(DFAProcessor):
#     def __init__(
#             self,
#             out_size: int,
#             mid_size: Optional[int] = None,
#             mid_act: Optional[_Fn] = None,
#             activation: Optional[_Fn] = jax.nn.relu,
#             reduction: _Fn = jnp.max,
#             msgs_mlp_sizes: Optional[List[int]] = None,
#             use_ln: bool = False,
#             use_triplets: bool = False,
#             nb_triplet_fts: int = 8,
#             gated: bool = False,
#             name: str = 'mpnn_aggr',
#     ):
#         super().__init__(name=name)
#         if mid_size is None:
#             self.mid_size = out_size
#         else:
#             self.mid_size = mid_size
#         self.out_size = out_size
#         self.mid_act = mid_act
#         self.activation = activation
#         self.reduction = reduction
#         self._msgs_mlp_sizes = msgs_mlp_sizes
#         self.use_ln = use_ln
#         self.use_triplets = use_triplets
#         self.nb_triplet_fts = nb_triplet_fts
#         self.gated = gated
#
#     def __call__(self,
#                  node_fts: _chex_Array,  # [N, hidden_dim]
#                  gkt_edge_fts: _chex_Array,  # [E_gkt, hidden_dim]
#                  hidden: _chex_Array,  # [N, hidden_dim]
#                  cfg_indices_padded: _chex_Array,  # [E_cfg, 2]
#                  gkt_indices_padded: _chex_Array,  # [E_gkt, 2]
#                  ):
#         """MPNN inference step."""
#
#         nb_nodes_padded = node_fts.shape[0]
#         nb_cfg_edges_padded, nb_gkt_edges_padded = cfg_indices_padded.shape[0], gkt_indices_padded.shape[0]
#
#         z = jnp.concatenate([node_fts, hidden], axis=-1)  # [N, 2 * hidden_dim]
#         m_1 = hk.Linear(self.mid_size)
#         m_2 = hk.Linear(self.mid_size)
#         m_e = hk.Linear(self.mid_size)
#         m_g = hk.Linear(self.mid_size)
#
#         msg_1 = m_1(z)  # [N, mid_size]
#         msg_2 = m_2(z)  # [N, mid_size]
#         msg_e = m_e(edge_fts)  # [E, mide_size]
#         msg_g = m_g(graph_fts)  # [B, mide_size]
#
#         tri_msgs = None
#
#         if self.use_triplets:
#             # Triplet messages, as done by Dudzik and Velickovic (2022)
#             triplets = get_triplet_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)
#
#             o3 = hk.Linear(self.out_size)
#             tri_msgs = o3(jnp.max(triplets, axis=1))  # (B, N, N, H)
#
#             if self.activation is not None:
#                 tri_msgs = self.activation(tri_msgs)
#
#         # msgs = (
#         #         jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
#         #         msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))  # [B, N, N, mide_size]
#         msgs = (msg_1[row_indices] +  # [E, mide_size]
#                 msg_2[col_indices] +  # [E, mide_size]
#                 msg_e +
#                 jnp.repeat(a=msg_g,
#                            repeats=nb_edges_each_graph, axis=0))  # [E, mide_size]
#
#         if self._msgs_mlp_sizes is not None:
#             msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))
#
#         if self.mid_act is not None:
#             msgs = self.mid_act(msgs)
#
#         if self.reduction == jnp.mean:
#             # adj_mat.shape = [B, N, N]
#             # msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)  # [B, N, H]
#             # msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
#             msgs = jax.ops.segment_sum(data=msgs,
#                                        segment_ids=row_indices,
#                                        num_segments=nb_nodes)  # [N, mid_size]
#             d = jax.ops.segment_sum(data=np.ones_like(col_indices),
#                                     segment_ids=col_indices,
#                                     num_segments=nb_nodes)  # [N, ]
#             msgs = msgs / jnp.expand_dims(d, axis=-1)
#
#         elif self.reduction == jnp.max:
#             # maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
#             #                    msgs,
#             #                    -BIG_NUMBER)     # [B, N, N, H]
#             # msgs = jnp.max(maxarg, axis=1)
#             msgs = jax.ops.segment_max(data=msgs,
#                                        segment_ids=col_indices,
#                                        num_segments=nb_nodes)  # [N, mid_size]
#         else:
#             raise Exception('Unrecognized reduction!')
#
#         o1 = hk.Linear(self.out_size)
#         o2 = hk.Linear(self.out_size)
#         h_1 = o1(z)  # [N, out_size]
#         h_2 = o2(msgs)  # [N, out_size]
#
#         ret = h_1 + h_2  # [N, out_size]
#
#         if self.activation is not None:
#             ret = self.activation(ret)
#
#         if self.use_ln:
#             ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
#             ret = ln(ret)
#
#         if self.gated:
#             gate1 = hk.Linear(self.out_size)
#             gate2 = hk.Linear(self.out_size)
#             gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))
#             gate = jax.nn.sigmoid(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
#             ret = ret * gate + hidden * (1 - gate)
#
#         return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars
#
#
# def get_triplet_msgs_sparse(z,  # [N, 2 * hidden_dim]
#                             edge_fts,  # [E, hidden_dim]
#                             graph_fts,  # [B, hidden_dim]
#                             row_indices,  # [E, ]
#                             col_indices,  # [E, ]
#                             nb_triplet_fts):
#     """Triplet messages, as done by Dudzik and Velickovic (2022). sparse version"""
#     t_1 = hk.Linear(nb_triplet_fts)
#     t_2 = hk.Linear(nb_triplet_fts)
#     t_3 = hk.Linear(nb_triplet_fts)
#     t_e_1 = hk.Linear(nb_triplet_fts)
#     t_e_2 = hk.Linear(nb_triplet_fts)
#     t_e_3 = hk.Linear(nb_triplet_fts)
#     t_g = hk.Linear(nb_triplet_fts)
#
#     tri_1 = t_1(z)  # [N, ntf]
#     tri_2 = t_2(z)
#     tri_3 = t_3(z)
#     tri_e_1 = t_e_1(edge_fts)
#     tri_e_2 = t_e_2(edge_fts)
#     tri_e_3 = t_e_3(edge_fts)
#     tri_g = t_g(graph_fts)
#
#     return (
#             jnp.expand_dims(tri_1, axis=(2, 3)) +  # (B, N, 1, 1, H)
#             jnp.expand_dims(tri_2, axis=(1, 3)) +  # + (B, 1, N, 1, H)
#             jnp.expand_dims(tri_3, axis=(1, 2)) +  # + (B, 1, 1, N, H)
#             jnp.expand_dims(tri_e_1, axis=3) +  # + (B, N, N, 1, H)
#             jnp.expand_dims(tri_e_2, axis=2) +  # + (B, N, 1, N, H)
#             jnp.expand_dims(tri_e_3, axis=1) +  # + (B, 1, N, N, H)
#             jnp.expand_dims(tri_g, axis=(1, 2, 3))  # + (B, 1, 1, 1, H)
#     )


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
            processor = GATSparse(out_size=out_size,
                                  nb_heads=nb_heads,
                                  activation=activation,
                                  residual=residual,
                                  use_ln=use_ln)
        else:
            raise ValueError('Unexpected processor kind ' + kind)

        return processor

    return _dfa_factory
