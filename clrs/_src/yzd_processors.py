import jax.ops
import numpy as np

from clrs._src.processors import *

_Array = chex.Array


class GATSparse(GAT):
    def __call__(  # pytype: disable=signature-mismatch  # numpy-scalars
            self,
            node_fts: _Array,  # [N, hidden_dim]
            edge_fts: _Array,  # [E, hidden_dim]
            graph_fts: _Array,  # [B, hidden_dim]
            # adj_mat: _Array,
            hidden: _Array,
            row_indices,  # [E, ]
            col_indices,
            nb_edges_each_graph,  # [B, ]
    ):
        """GAT inference step (sparse version by yzd)."""

        batch_size, _ = graph_fts.shape
        nb_nodes = node_fts.shape[0]
        nb_edges = row_indices.shape[0]
        assert nb_edges_each_graph.shape[0] == batch_size
        assert col_indices.shape[0] == nb_edges
        assert jnp.sum(nb_edges_each_graph) == nb_edges
        # assert graph_fts.shape[:-1] == (b,)
        # assert adj_mat.shape == (b, n, n)

        z = jnp.concatenate([node_fts, hidden], axis=-1)  # [N, 2*hidden_dim]
        m = hk.Linear(self.out_size)
        skip = hk.Linear(self.out_size)

        # bias_mat = (adj_mat - 1.0) * 1e9
        # bias_mat = jnp.tile(bias_mat[..., None],
        #                     (1, 1, 1, self.nb_heads))  # [B, N, N, H]
        # bias_mat = jnp.transpose(bias_mat, (0, 3, 1, 2))  # [B, H, N, N]

        a_1 = hk.Linear(self.nb_heads)
        a_2 = hk.Linear(self.nb_heads)
        a_e = hk.Linear(self.nb_heads)
        a_g = hk.Linear(self.nb_heads)
        att_1 = a_1(z)  # [N, H]
        att_2 = a_2(z)  # [N, H]
        att_e = a_e(edge_fts)  # [E, H]
        att_g = a_g(graph_fts)  # [B, H]
        keys_source = att_1[row_indices]  # [E, H]
        queries_dest = att_2[col_indices]  # [E, H]
        logits = (
                att_1[row_indices] +  # + [E, H]
                att_2[col_indices] +  # + [E, H]
                att_e +  # + [E, H]
                jnp.repeat(a=att_g, repeats=nb_edges_each_graph, axis=0)  # + [E, H]
        )  # = [E, H]
        coefs = unsorted_segment_softmax(logits=jax.nn.leaky_relu(logits),
                                         segment_ids=col_indices,
                                         num_segments=nb_edges)

        values = m(z)  # [N, H*F]
        values = jnp.reshape(
            values,
            values.shape[:-1] + (self.nb_heads, self.head_size))  # [N, H, F]
        values_source = values[col_indices]  # [E, H, F]

        ret = coefs * values_source  # [E, H, F]
        ret = jax.ops.segment_sum(data=ret,
                                  segment_ids=row_indices,
                                  num_segments=nb_nodes)  # [N, H, F]
        ret = jnp.reshape(ret, ret.shape[:-2] + (self.out_size,))  # [N, H*F]

        if self.residual:
            ret += skip(z)

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
