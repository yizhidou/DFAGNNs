import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.ops
import numpy as np

# from clrs._src.processors import *

import abc
from typing import Any, Callable, List, Optional, Tuple

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
            gkt_edge_fts: _chex_Array,
            hidden: _chex_Array,
            cfg_indices_padded: _chex_Array,
            gkt_indices_padded: _chex_Array,
    ) -> Tuple[_chex_Array, Optional[_chex_Array]]:
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
            name: str = 'gat_aggr',
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
            node_fts: _chex_Array,  # [N, hidden_dim]
            gkt_edge_fts: _chex_Array,  # [E_gkt, hidden_dim]
            hidden: _chex_Array,  # [N, hidden_dim]
            cfg_indices_padded: _chex_Array,  # [E_cfg, 2]
            gkt_indices_padded: _chex_Array,  # [E_gkt, 2]
    ):
        """GAT inference step (sparse version by yzd)."""
        nb_nodes_padded = node_fts.shape[0]
        nb_cfg_edges_padded, nb_gkt_edges_padded = cfg_indices_padded.shape[0], gkt_indices_padded.shape[0]
        cfg_row_indices = cfg_indices_padded[:, 0]  # [E_cfg, ]
        cfg_col_indices = cfg_indices_padded[:, 1]
        gkt_row_indices = gkt_indices_padded[:, 0]  # [E_gkt, ]
        gkt_col_indices = gkt_indices_padded[:, 1]

        cfg_z = jnp.concatenate([node_fts, hidden], axis=-1)  # [N, 2*hidden_dim]
        m = hk.Linear(self.out_size)
        skip = hk.Linear(self.out_size)

        a_1 = hk.Linear(self.nb_heads)
        a_2 = hk.Linear(self.nb_heads)
        a_e = hk.Linear(self.nb_heads)
        cfg_att_1 = a_1(cfg_z)  # [N, H]
        cfg_att_2 = a_2(cfg_z)  # [N, H]

        cfg_logits = (
                cfg_att_1[cfg_row_indices] +  # + [E_cfg, H]
                cfg_att_2[cfg_col_indices]  # + [E_cfg, H]
        )  # = [E_cfg, H]
        cfg_coefs = unsorted_segment_softmax(logits=jax.nn.leaky_relu(cfg_logits),
                                             segment_ids=cfg_row_indices,
                                             num_segments=nb_cfg_edges_padded)
        # [E_cfg, H]

        cfg_values = m(cfg_z)  # [N, H*F]
        cfg_values = jnp.reshape(
            cfg_values,
            cfg_values.shape[:-1] + (self.nb_heads, self.head_size))  # [N, H, F]
        cfg_values_source = cfg_values[cfg_col_indices]  # [E_cfg, H, F]

        cfg_hidden = jnp.expand_dims(cfg_coefs, axis=-1) * cfg_values_source  # [E_cfg, H, F]
        cfg_hidden = jax.ops.segment_sum(data=cfg_hidden,
                                         segment_ids=cfg_row_indices,
                                         num_segments=nb_nodes_padded)  # [N, H, F]
        cfg_hidden = jnp.reshape(cfg_hidden, cfg_hidden.shape[:-2] + (self.out_size,))  # [N, H*F]

        if self.residual:
            cfg_hidden += skip(cfg_z)

        if self.activation is not None:
            cfg_hidden = self.activation(cfg_hidden)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            cfg_hidden = ln(cfg_hidden)

        gkt_z = jnp.concatenate([node_fts, cfg_hidden], axis=-1)  # [N, 2*hidden_dim]
        gkt_att_1 = a_1(gkt_z)  # [N, H]
        gkt_att_2 = a_2(gkt_z)  # [N, H]
        gkt_att_e = a_e(gkt_edge_fts)  # [E_gkt, H]
        # NOT SURE att_g
        # att_g = a_g(graph_fts)  # [B, H]
        gkt_logits = (
                gkt_att_1[gkt_row_indices] +  # + [E_gkt, H]
                gkt_att_2[gkt_col_indices] +  # + [E_gkt, H]
                gkt_att_e  # + [E_gkt, H]  # + [E, H]
        )  # = [E_gkt, H]
        gkt_coefs = unsorted_segment_softmax(logits=jax.nn.leaky_relu(gkt_logits),
                                             segment_ids=gkt_row_indices,
                                             num_segments=nb_gkt_edges_padded)

        gkt_values = m(gkt_z)  # [N, H*F]
        gkt_values = jnp.reshape(
            gkt_values,
            gkt_values.shape[:-1] + (self.nb_heads, self.head_size))  # [N, H, F]
        gkt_values_source = gkt_values[gkt_col_indices]  # [E_gkt, H, F]

        ret = jnp.expand_dims(gkt_coefs, axis=-1) * gkt_values_source  # [E_gkt, H, F]
        ret = jax.ops.segment_sum(data=ret,
                                  segment_ids=gkt_row_indices,
                                  num_segments=nb_nodes_padded)  # [N, H, F]
        ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [N, H*F]

        if self.residual:
            ret += skip(gkt_z)

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        return ret, None  # pytype: disable=bad-return-type  # numpy-scalars


def unsorted_segment_softmax(logits, segment_ids, num_segments):
    """Returns softmax over each segment.

    Args:
      logits: Logits of elements, of shape `[dim_0, ...]`.
      segment_ids: Segmentation of logits, which elements will participate in
        the same softmax. Shape `[dim_0]`.
      num_segments: Scalar number of segments, typically `max(segment_ids)`.

    Returns:
      Probabilities of the softmax, shape `[dim_0, ...]`.
    """
    segment_max_ = jax.ops.segment_max(logits, segment_ids, num_segments)
    broadcast_segment_max = segment_max_[segment_ids]
    shifted_logits = logits - broadcast_segment_max

    # Sum and get the probabilities.
    exp_logits = jnp.exp(shifted_logits)
    exp_sum = jax.ops.segment_sum(exp_logits, segment_ids, num_segments)
    broadcast_exp_sum = exp_sum[segment_ids]
    probs = exp_logits / broadcast_exp_sum
    return probs


class PGNSparse(PGN):
    def __call__(self,
                 node_fts: _Array,  # [N, hidden_dim]
                 edge_fts: _Array,  # [E, hidden_dim]
                 graph_fts: _Array,  # [B, hidden_dim]
                 # adj_mat: _Array,
                 hidden: _Array,
                 row_indices,  # [E, ]
                 col_indices,
                 nb_edges_each_graph,  # [B, ]
                 ):
        """MPNN inference step."""

        batch_size, _ = graph_fts.shape
        nb_nodes, _ = node_fts.shape
        nb_edges = edge_fts.shape[0]
        assert nb_edges_each_graph.shape[0] == batch_size
        assert jnp.sum(nb_edges_each_graph) == nb_edges
        # b, n, _ = node_fts.shape
        # assert edge_fts.shape[:-1] == (b, n, n)
        # assert graph_fts.shape[:-1] == (b,)
        # assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)  # [N, 2 * hidden_dim]
        m_1 = hk.Linear(self.mid_size)
        m_2 = hk.Linear(self.mid_size)
        m_e = hk.Linear(self.mid_size)
        m_g = hk.Linear(self.mid_size)

        msg_1 = m_1(z)  # [N, mide_size]
        msg_2 = m_2(z)  # [N, mide_size]
        msg_e = m_e(edge_fts)  # [E, mide_size]
        msg_g = m_g(graph_fts)  # [B, mide_size]

        tri_msgs = None

        if self.use_triplets:
            # Triplet messages, as done by Dudzik and Velickovic (2022)
            triplets = get_triplet_msgs(z, edge_fts, graph_fts, self.nb_triplet_fts)

            o3 = hk.Linear(self.out_size)
            tri_msgs = o3(jnp.max(triplets, axis=1))  # (B, N, N, H)

            if self.activation is not None:
                tri_msgs = self.activation(tri_msgs)

        # msgs = (
        #         jnp.expand_dims(msg_1, axis=1) + jnp.expand_dims(msg_2, axis=2) +
        #         msg_e + jnp.expand_dims(msg_g, axis=(1, 2)))  # [B, N, N, mide_size]
        msgs = (msg_1[row_indices] +  # [E, mide_size]
                msg_2[col_indices] +  # [E, mide_size]
                msg_e +
                jnp.repeat(a=msg_g,
                           repeats=nb_edges_each_graph, axis=0))  # [E, mide_size]

        if self._msgs_mlp_sizes is not None:
            msgs = hk.nets.MLP(self._msgs_mlp_sizes)(jax.nn.relu(msgs))

        if self.mid_act is not None:
            msgs = self.mid_act(msgs)

        if self.reduction == jnp.mean:
            # adj_mat.shape = [B, N, N]
            # msgs = jnp.sum(msgs * jnp.expand_dims(adj_mat, -1), axis=1)  # [B, N, H]
            # msgs = msgs / jnp.sum(adj_mat, axis=-1, keepdims=True)
            msgs = jax.ops.segment_sum(data=msgs,
                                       segment_ids=row_indices,
                                       num_segments=nb_nodes)  # [N, mid_size]
            d = jax.ops.segment_sum(data=np.ones_like(col_indices),
                                    segment_ids=col_indices,
                                    num_segments=nb_nodes)  # [N, ]
            msgs = msgs / jnp.expand_dims(d, axis=-1)

        elif self.reduction == jnp.max:
            # maxarg = jnp.where(jnp.expand_dims(adj_mat, -1),
            #                    msgs,
            #                    -BIG_NUMBER)     # [B, N, N, H]
            # msgs = jnp.max(maxarg, axis=1)
            msgs = jax.ops.segment_max(data=msgs,
                                       segment_ids=col_indices,
                                       num_segments=nb_nodes)  # [N, mid_size]
        else:
            raise Exception('Unrecognized reduction!')

        o1 = hk.Linear(self.out_size)
        o2 = hk.Linear(self.out_size)
        h_1 = o1(z)  # [N, out_size]
        h_2 = o2(msgs)  # [N, out_size]

        ret = h_1 + h_2  # [N, out_size]

        if self.activation is not None:
            ret = self.activation(ret)

        if self.use_ln:
            ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
            ret = ln(ret)

        if self.gated:
            gate1 = hk.Linear(self.out_size)
            gate2 = hk.Linear(self.out_size)
            gate3 = hk.Linear(self.out_size, b_init=hk.initializers.Constant(-3))
            gate = jax.nn.sigmoid(gate3(jax.nn.relu(gate1(z) + gate2(msgs))))
            ret = ret * gate + hidden * (1 - gate)

        return ret, tri_msgs  # pytype: disable=bad-return-type  # numpy-scalars


def get_triplet_msgs_sparse(z,  # [N, 2 * hidden_dim]
                            edge_fts,  # [E, hidden_dim]
                            graph_fts,  # [B, hidden_dim]
                            row_indices,  # [E, ]
                            col_indices,  # [E, ]
                            nb_triplet_fts):
    """Triplet messages, as done by Dudzik and Velickovic (2022). sparse version"""
    t_1 = hk.Linear(nb_triplet_fts)
    t_2 = hk.Linear(nb_triplet_fts)
    t_3 = hk.Linear(nb_triplet_fts)
    t_e_1 = hk.Linear(nb_triplet_fts)
    t_e_2 = hk.Linear(nb_triplet_fts)
    t_e_3 = hk.Linear(nb_triplet_fts)
    t_g = hk.Linear(nb_triplet_fts)

    tri_1 = t_1(z)  # [N, ntf]
    tri_2 = t_2(z)
    tri_3 = t_3(z)
    tri_e_1 = t_e_1(edge_fts)
    tri_e_2 = t_e_2(edge_fts)
    tri_e_3 = t_e_3(edge_fts)
    tri_g = t_g(graph_fts)

    return (
            jnp.expand_dims(tri_1, axis=(2, 3)) +  # (B, N, 1, 1, H)
            jnp.expand_dims(tri_2, axis=(1, 3)) +  # + (B, 1, N, 1, H)
            jnp.expand_dims(tri_3, axis=(1, 2)) +  # + (B, 1, 1, N, H)
            jnp.expand_dims(tri_e_1, axis=3) +  # + (B, N, N, 1, H)
            jnp.expand_dims(tri_e_2, axis=2) +  # + (B, N, 1, N, H)
            jnp.expand_dims(tri_e_3, axis=1) +  # + (B, 1, N, N, H)
            jnp.expand_dims(tri_g, axis=(1, 2, 3))  # + (B, 1, 1, 1, H)
    )


DFAProcessorFactory = Callable[[int], DFAProcessor]


def get_processor_factory(kind: str,
                          use_ln: bool,
                          nb_triplet_fts: int,
                          nb_heads: Optional[int] = None) -> DFAProcessorFactory:
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
        if kind == 'gat_dfa':
            processor = GATSparse(
                out_size=out_size,
                nb_heads=nb_heads,
                use_ln=use_ln,
            )
        else:
            raise ValueError('Unexpected processor kind ' + kind)

        return processor

    return _factory
