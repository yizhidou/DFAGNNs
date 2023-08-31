import functools

from typing import Dict, List, Optional, Tuple, Union

import chex

from clrs._src import decoders
from clrs._src import encoders, yzd_encoders
from clrs._src import probing, yzd_probing
from clrs._src import processors
from clrs._src import samplers, yzd_samplers
from clrs._src import specs, yzd_specs
from clrs._src import nets

import haiku as hk
import jax
import jax.numpy as jnp

_chex_Array = chex.Array
_ArraySparse = yzd_probing.ArraySparse
_DataPoint = yzd_probing.DataPoint
_Features = yzd_samplers.Features
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = yzd_samplers.Trajectory
_Type = specs.Type

_MessagePassingScanState = nets._MessagePassingScanState


class YZDNet(nets.Net):
    """Building blocks (networks) used to encode and decode messages."""

    def __call__(self, features_list: List[_Features], repred: bool,
                 algorithm_index: int,
                 return_hints: bool,
                 return_all_outputs: bool):
        """Process one batch of data.

        Args:
          features_list: A list of _Features objects, each with the inputs, hints
            and lengths for a batch o data corresponding to one algorithm.
            The list should have either length 1, at train/evaluation time,
            or length equal to the number of algorithms this Net is meant to
            process, at initialization.
          repred: False during training, when we have access to ground-truth hints.
            True in validation/test mode, when we have to use our own
            hint predictions.
          algorithm_index: Which algorithm is being processed. It can be -1 at
            initialisation (either because we are initialising the parameters of
            the module or because we are intialising the message-passing state),
            meaning that all algorithms should be processed, in which case
            `features_list` should have length equal to the number of specs of
            the Net. Otherwise, `algorithm_index` should be
            between 0 and `length(self.spec) - 1`, meaning only one of the
            algorithms will be processed, and `features_list` should have length 1.
          return_hints: Whether to accumulate and return the predicted hints,
            when they are decoded.
          return_all_outputs: Whether to return the full sequence of outputs, or
            just the last step's output.

        Returns:
          A 2-tuple with (output predictions, hint predictions)
          for the selected algorithm.
        """
        if algorithm_index == -1:
            algorithm_indices = range(len(features_list))
        else:
            algorithm_indices = [algorithm_index]
        assert len(algorithm_indices) == len(features_list)

        self.encoders, self.decoders = self._construct_encoders_decoders()
        self.processor = self.processor_factory(self.hidden_dim)

        # Optionally construct LSTM.
        if self.use_lstm:
            self.lstm = hk.LSTM(
                hidden_size=self.hidden_dim,
                name='processor_lstm')
            lstm_init = self.lstm.initial_state
        else:
            self.lstm = None
            lstm_init = lambda x: 0

        for algorithm_index, features in zip(algorithm_indices, features_list):
            input_NODE_dp_list = features.input_NODE_dp_list
            input_EDGE_dp_list = features.input_EDGE_dp_list
            trace_h = features.trace_h
            hint_len = features.hint_len

            batch_size, nb_nodes_entire_batch = _data_dimensions(features)

            # YZDTODO 不知道为什么要减一 但是先不管了
            nb_mp_steps = max(1, trace_h.data.edge_indices_with_optional_content.shape[0] - 1)
            hiddens = jnp.zeros((batch_size, nb_nodes_entire_batch, self.hidden_dim))

            if self.use_lstm:
                lstm_state = lstm_init(batch_size * nb_nodes_entire_batch)
                lstm_state = jax.tree_util.tree_map(
                    lambda x, b=batch_size, n=nb_nodes_entire_batch: jnp.reshape(x, [b, n, -1]),
                    lstm_state)
            else:
                lstm_state = None

            mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
                hint_preds=None,
                output_preds=None,
                hiddens=hiddens,
                lstm_state=lstm_state)

            # Do the first step outside of the scan because it has a different
            # computation graph.
            common_args = dict(
                hints=hints,
                repred=repred,
                inputs=inputs,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
                lengths=lengths,
                spec=self.spec[algorithm_index],
                encs=self.encoders[algorithm_index],
                decs=self.decoders[algorithm_index],
                return_hints=return_hints,
                return_all_outputs=return_all_outputs,
            )
            mp_state, lean_mp_state = self._msg_passing_step(
                mp_state,
                i=0,
                first_step=True,
                **common_args)

            # Then scan through the rest.
            scan_fn = functools.partial(
                self._msg_passing_step,
                first_step=False,
                **common_args)

            output_mp_state, accum_mp_state = hk.scan(
                scan_fn,
                mp_state,
                jnp.arange(nb_mp_steps - 1) + 1,
                length=nb_mp_steps - 1)

        # We only return the last algorithm's output. That's because
        # the output only matters when a single algorithm is processed; the case
        # `algorithm_index==-1` (meaning all algorithms should be processed)
        # is used only to init parameters.
        accum_mp_state = jax.tree_util.tree_map(
            lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
            lean_mp_state, accum_mp_state)

        def invert(d):
            """Dict of lists -> list of dicts."""
            if d:
                return [dict(zip(d, i)) for i in zip(*d.values())]

        if return_all_outputs:
            output_preds = {k: jnp.stack(v)
                            for k, v in accum_mp_state.output_preds.items()}
        else:
            output_preds = output_mp_state.output_preds
        hint_preds = invert(accum_mp_state.hint_preds)

        return output_preds, hint_preds

    def _msg_passing_step(self,
                          mp_state: _MessagePassingScanState,
                          i: int,
                          # dense_hints: List[_DataPoint],
                          trace_h: _DataPoint,
                          repred: bool,
                          lengths: chex.Array,
                          batch_size: int,
                          nb_nodes: int,
                          input_NODE_dp_list: _Trajectory,
                          input_EDGE_dp_list: _Trajectory,
                          first_step: bool,
                          spec: _Spec,
                          encs: Dict[str, List[hk.Module]],
                          decs: Dict[str, Tuple[hk.Module]],
                          return_hints: bool,
                          return_all_outputs: bool
                          ):
        if self.decode_hints and not first_step:
            assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
            hard_postprocess = (self._hint_repred_mode == 'hard' or
                                (self._hint_repred_mode == 'hard_on_eval' and repred))
            decoded_hint = decoders.postprocess(spec,
                                                mp_state.hint_preds,
                                                sinkhorn_temperature=0.1,
                                                sinkhorn_steps=25,
                                                hard=hard_postprocess)
        if repred and self.decode_hints and not first_step:
            cur_hint = []
            for hint_name in decoded_hint:
                cur_hint.append(decoded_hint[hint_name])
        else:
            # cur_trace_h = []
            needs_noise = (self.decode_hints and not first_step and
                           self._hint_teacher_forcing < 1.0)

            if needs_noise:
                # For noisy teacher forcing, choose which examples in the batch to force
                force_mask = jax.random.bernoulli(
                    hk.next_rng_key(), self._hint_teacher_forcing,
                    (batch_size,))
            else:
                force_mask = None
            # 我其实不确定这地方的数据类型应该是什么，但先这样写吧
            trace_h_i_data = _ArraySparse(
                edges_with_optional_content=jnp.asarray(trace_h.data.edges_with_optional_content)[i],
                nb_nodes=jnp.asarray(trace_h.data.nb_nodes),
                nb_edges=jnp.asarray(trace_h.data.nb_edges))
            if needs_noise:
                pass
            trace_h_i = _DataPoint(name=trace_h.name,
                                     location=trace_h.location,
                                     type_=trace_h.type_,
                                     data=trace_h_i_data)

        hiddens, output_preds_cand, hint_preds, lstm_state = self._one_step_pred(
            inputs, trace_h_i, mp_state.hiddens,
            batch_size, nb_nodes, mp_state.lstm_state,
            spec, encs, decs, repred)

        if first_step:
            output_preds = output_preds_cand
        else:
            output_preds = {}
            for outp in mp_state.output_preds:
                is_not_done = _is_not_done_broadcast(lengths, i,
                                                     output_preds_cand[outp])
                output_preds[outp] = is_not_done * output_preds_cand[outp] + (
                        1.0 - is_not_done) * mp_state.output_preds[outp]

        new_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            hint_preds=hint_preds,
            output_preds=output_preds,
            hiddens=hiddens,
            lstm_state=lstm_state)
        # Save memory by not stacking unnecessary fields
        accum_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            hint_preds=hint_preds if return_hints else None,
            output_preds=output_preds if return_all_outputs else None,
            hiddens=None, lstm_state=None)

        # Complying to jax.scan, the first returned value is the state we carry over
        # the second value is the output that will be stacked over steps.
        return new_mp_state, accum_mp_state

    def _one_step_pred(
            self,
            input_NODE_dp_list: _Trajectory,
            input_EDGE_dp_list: _Trajectory,
            trace_h_i: _DataPoint,
            hidden: _chex_Array,
            batch_size: int,
            # nb_nodes_entire_batch: int,
            lstm_state: Optional[hk.LSTMState],
            spec: _Spec,
            encs: Dict[str, List[hk.Module]],
            decs: Dict[str, Tuple[hk.Module]],
            repred: bool,
    ):
        """Generates one-step predictions."""

        # Initialise empty node/edge/graph features and adjacency matrix.

        # graph_fts = jnp.zeros((batch_size, self.hidden_dim))
        # adj_mat = jnp.repeat(
        #     jnp.expand_dims(jnp.eye(nb_nodes), 0), batch_size, axis=0)

        info_dict = yzd_encoders.func(input_NODE_dp_list=input_NODE_dp_list,
                                      input_EDGE_dp_list=input_EDGE_dp_list,
                                      trace_h_i=trace_h_i)
        nb_nodes_entire_batch = jnp.sum(info_dict['nb_nodes']).item()
        nb_edges_entire_batch = jnp.sum(info_dict['nb_kgt_edges']).item()
        node_fts = jnp.zeros((nb_nodes_entire_batch, self.hidden_dim))
        kgt_edge_fts = jnp.zeros((nb_edges_entire_batch, self.hidden_dim))

        # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Encode node/edge/graph features from inputs and (optionally) hints.
        # encode node fts

        dense_trajectories = [dense_inputs]
        if self.encode_hints:
            dense_trajectories.append(dense_hints)

        for name in ['pos', 'if_pp', 'if_ip']:
            dp = info_dict[name]
            
        for dense_trajectory in dense_trajectories:
            for dp in dense_trajectory:
                try:
                    dp = encoders.preprocess(dp, nb_nodes)
                    assert dp.type_ != _Type.SOFT_POINTER
                    adj_mat = encoders.accum_adj_mat(dp, adj_mat)
                    encoder = encs[dp.name]
                    edge_fts = encoders.accum_edge_fts(encoder, dp, edge_fts)
                    node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
                    graph_fts = encoders.accum_graph_fts(encoder, dp, graph_fts)
                except Exception as e:
                    raise Exception(f'Failed to process {dp}') from e

        # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        nxt_hidden = hidden
        for _ in range(self.nb_msg_passing_steps):
            nxt_hidden, nxt_edge = self.processor(
                node_fts,
                edge_fts,
                graph_fts,
                adj_mat,
                nxt_hidden,
                batch_size=batch_size,
                nb_nodes=nb_nodes,
            )

        if not repred:  # dropout only on training
            nxt_hidden = hk.dropout(hk.next_rng_key(), self._dropout_prob, nxt_hidden)

        if self.use_lstm:
            # lstm doesn't accept multiple batch dimensions (in our case, batch and
            # nodes), so we vmap over the (first) batch dimension.
            nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(nxt_hidden, lstm_state)
        else:
            nxt_lstm_state = None

        h_t = jnp.concatenate([node_fts, hidden, nxt_hidden], axis=-1)
        if nxt_edge is not None:
            e_t = jnp.concatenate([edge_fts, nxt_edge], axis=-1)
        else:
            e_t = edge_fts

        # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decode features and (optionally) hints.
        hint_preds, output_preds = decoders.decode_fts(
            decoders=decs,
            spec=spec,
            h_t=h_t,
            adj_mat=adj_mat,
            edge_fts=e_t,
            graph_fts=graph_fts,
            inf_bias=self.processor.inf_bias,
            inf_bias_edge=self.processor.inf_bias_edge,
            repred=repred,
        )

        return nxt_hidden, output_preds, hint_preds, nxt_lstm_state


def _data_dimensions(features: _Features):
    """Returns (batch_size, nb_nodes)."""
    nb_nodes_entire_batch = None
    batch_size = None
    for input_NODE_dp in features.input_NODE_dp_list:
        if nb_nodes_entire_batch is None:
            nb_nodes_entire_batch = input_NODE_dp.shape[0]
        else:
            assert nb_nodes_entire_batch == input_NODE_dp.shape[0]
    for input_EDGE_dp in features.input_EDGE_dp_list:
        nb_nodes = input_EDGE_dp.data.nb_nodes
        assert jnp.sum(nb_nodes).item() == nb_nodes_entire_batch
        if batch_size is None:
            batch_size = nb_nodes.shape[0]
        else:
            assert batch_size == nb_nodes.shape[0]
    return batch_size, nb_nodes_entire_batch
