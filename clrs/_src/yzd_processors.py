import jax.ops

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
            source_indices,  # [E, ]
            dest_indices,
            nb_edges,  # [B, ]
            ):
        """GAT inference step (sparse version by yzd)."""

        batch_size, _ = graph_fts.shape
        assert nb_edges.shape[0] == batch_size
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
        keys_source = att_1[source_indices]  # [E, H]
        queries_dest = att_2[dest_indices]  # [E, H]
        logits = (
                att_1[source_indices] +  # + [E, H]
                att_2[dest_indices] +  # + [E, H]
                att_e +  # + [E, H]
                jnp.repeat(a=att_g, repeats=nb_edges, axis=0)  # + [E, H]
        )  # = [E, H]
        coefs = unsorted_segment_softmax(logits=jax.nn.leaky_relu(logits),
                                         segment_ids=dest_indices,
                                         num_segments=batch_size)

        values = m(z)  # [N, H*F]
        values = jnp.reshape(
            values,
            values.shape[:-1] + (self.nb_heads, self.head_size))  # [N, H, F]
        values_source = values[dest_indices]  # [E, H, F]

        ret = coefs * values_source  # [E, H, F]
        ret = jax.ops.segment_sum(data=ret,
                                  segment_ids=source_indices,
                                  num_segments=batch_size)  # [N, H, F]
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
