import functools

from typing import Dict, List, Optional, Tuple, Union

import chex

from clrs._src import decoders, yzd_decoders
from clrs._src import encoders, yzd_encoders
from clrs._src import probing, yzd_probing
from clrs._src import yzd_processors
from clrs._src import samplers, dfa_sampler
from clrs._src import specs, yzd_specs
from clrs._src import nets

import haiku as hk
import jax
import jax.numpy as jnp

_chex_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = dfa_sampler.Features
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = samplers.Trajectory
_Type = specs.Type


@chex.dataclass
class _MessagePassingScanState:
    pred_trace_h_i: _chex_Array
    pred_trace_o: _chex_Array
    hiddens: chex.Array
    lstm_state: Optional[hk.LSTMState]


class YZDNet(nets.Net):
    """Building blocks (networks) used to encode and decode messages."""

    def __init__(
            self,
            spec: List[_Spec],
            hidden_dim: int,
            encode_hints: bool,
            decode_hints: bool,
            processor_factory: yzd_processors.ProcessorFactory,
            use_lstm: bool,
            encoder_init: str,
            dropout_prob: float,
            hint_teacher_forcing: float,
            hint_repred_mode='soft',
            nb_dims=None,
            nb_msg_passing_steps=1,
            name: str = 'net',
    ):
        """Constructs a `Net`."""
        super().__init__(spec=spec,
                         hidden_dim=hidden_dim,
                         encode_hints=encode_hints,
                         decode_hints=decode_hints,
                         processor_factory=processor_factory,
                         use_lstm=use_lstm,
                         encoder_init=encoder_init,
                         dropout_prob=dropout_prob,
                         hint_teacher_forcing=hint_teacher_forcing,
                         hint_repred_mode=hint_repred_mode,
                         nb_dims=nb_dims,
                         nb_msg_passing_steps=nb_msg_passing_steps,
                         name=name)

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
            input_dp_list = features.input_dp_list
            trace_h = features.trace_h
            mask_dict = features.mask_dict

            batch_size, nb_nodes = nets._data_dimensions(features)

            # YZDTODO 不知道为什么要减一 但是先不管了
            nb_mp_steps = max(1, trace_h.data.shape[0] - 1)
            hiddens = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))

            if self.use_lstm:
                lstm_state = lstm_init(batch_size * nb_nodes)
                lstm_state = jax.tree_util.tree_map(
                    lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, -1]),
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

    def dfa_msg_passing_step(self,
                          mp_state: _MessagePassingScanState,
                          i: int,
                          # dense_hints: List[_DataPoint],
                          trace_h: _DataPoint,
                          repred: bool,
                          lengths: _chex_Array,
                          batch_size: int,
                          nb_nodes: int,
                          nb_gkt_edges: _chex_Array,
                          input_NODE_dp_list: _Trajectory,
                          input_EDGE_dp_list: _Trajectory,
                          first_step: bool,
                          spec: _Spec,
                          encs: Dict[str, List[hk.Module]],
                          decs: Dict[str, Tuple[hk.Module]],
                          return_hints: bool,
                          return_all_outputs: bool
                          ):
        trace_h_i = jnp.asarray(trace_h.data)[i]
        if self.decode_hints and not first_step:
            assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
            hard_postprocess = (self._hint_repred_mode == 'hard' or
                                (self._hint_repred_mode == 'hard_on_eval' and repred))
            if hard_postprocess:
                # 这地方不太对哈，最起码只能处理最后一位
                decoded_trace_h_i = (mp_state.pred_trace_h_i > 0.0) * 1.0
            else:
                decoded_trace_h_i = jax.nn.sigmoid(mp_state.pred_trace_h_i)
            if repred:
                trace_h_i = decoded_trace_h_i
            elif self._hint_teacher_forcing < 1.0:
                force_mask = jax.random.bernoulli(
                    hk.next_rng_key(), self._hint_teacher_forcing,
                    (batch_size,))
                # force_mask = jnp.repeat(force_mask, nb_gkt_edges)
                trace_h_i = jnp.where(nets._expand_to(force_mask, trace_h_i),
                                      trace_h_i,
                                      decoded_trace_h_i)

        hiddens, pred_trace_o_cand, hint_preds, lstm_state = self._one_step_pred(
            input_NODE_dp_list=input_NODE_dp_list,
            input_EDGE_dp_list=input_EDGE_dp_list,
            trace_h_i=trace_h_i,
            hidden=mp_state.hiddens,
            lstm_state=mp_state.lstm_state,
            encs=encs,
            decs=decs,
            repred=repred)

        if first_step:
            pred_trace_o = pred_trace_o_cand
        else:
            # output_preds = {}
            is_not_done = nets._is_not_done_broadcast(lengths, i,
                                                      pred_trace_o_cand)
            pred_trace_o = is_not_done * pred_trace_o_cand + (
                    1.0 - is_not_done) * mp_state.output_preds

        new_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            hint_preds=hint_preds,
            output_preds=pred_trace_o,
            hiddens=hiddens,
            lstm_state=lstm_state)
        # Save memory by not stacking unnecessary fields
        accum_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            hint_preds=hint_preds if return_hints else None,
            output_preds=pred_trace_o if return_all_outputs else None,
            hiddens=None, lstm_state=None)

        # Complying to jax.scan, the first returned value is the state we carry over
        # the second value is the output that will be stacked over steps.
        return new_mp_state, accum_mp_state

    def _one_step_pred(
            self,
            input_NODE_dp_list: _Trajectory,
            input_EDGE_dp_list: _Trajectory,
            trace_h_i: _chex_Array,
            hidden: _chex_Array,
            # batch_size: int,
            # nb_nodes_entire_batch: int,
            lstm_state: Optional[hk.LSTMState],
            # spec: _Spec,
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
        gkt_edge_fts = jnp.zeros((nb_edges_entire_batch, self.hidden_dim))

        # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Encode node/edge/graph features from inputs and (optionally) hints.
        # encode node fts
        for name in ['pos', 'if_pp', 'if_ip']:
            dp_content = info_dict[name]
            encoder = encs[name]
            encoding = encoder[0](dp_content)
            node_fts += encoding
        # encode edge fts
        for name in ['gen_sparse', 'kill_sparse', 'trace_i_sparse']:
            if name in info_dict:
                dp_content = info_dict[name]
            else:
                continue
            encoder = encs[name]
            encoding = encoder[0](dp_content)
            gkt_edge_fts += encoding
        # encode hints
        if self.encode_hints:
            encoder = encs['trace_h_sparse']
            encoding = encoder[0](info_dict['trace_h_i'])
            gkt_edge_fts += encoding

        # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        nxt_hidden = hidden
        for _ in range(self.nb_msg_passing_steps):
            nxt_hidden, nxt_edge = self.processor(
                node_fts=node_fts,
                gkt_edge_fts=gkt_edge_fts,
                hidden=nxt_hidden,
                cfg_edges=info_dict['cfg_edges'],
                nb_cfg_edges_each_graph=info_dict['nb_cfg_edges'],
                gkt_edges=info_dict['gkt_edges'],
                nb_gkt_edges_each_graph=info_dict['nb_gkt_edges']
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
            e_t = jnp.concatenate([gkt_edge_fts, nxt_edge], axis=-1)
        else:
            e_t = gkt_edge_fts

        # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decode features and (optionally) hints.
        pred_trace_o, pred_trace_h_i = yzd_decoders.decode_fts(
            decoders=decs,
            h_t=h_t,
            gkt_edge_fts=gkt_edge_fts,
            gkt_edges=info_dict['gkt_edges']
        )
