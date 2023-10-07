import functools

from typing import Dict, List, Optional, Tuple, Union

import chex

from clrs._src import decoders, dfa_decoders
from clrs._src import encoders
from clrs._src import probing
from clrs._src import dfa_processors
from clrs._src import samplers, dfa_sampler
from clrs._src import specs, yzd_specs
from clrs._src import nets

import haiku as hk
import jax
import jax.numpy as jnp

_chex_Array = chex.Array
# _DataPoint = probing.DataPoint
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
    hiddens: _chex_Array
    lstm_state: Optional[hk.LSTMState]


class DFANet(nets.Net):
    """Building blocks (networks) used to encode and decode messages."""

    def __init__(
            self,
            spec: List[_Spec],
            hidden_dim: int,
            encode_hints: bool,
            decode_hints: bool,
            processor_factory: dfa_processors.DFAProcessorFactory,
            use_lstm: bool,
            encoder_init: str,
            dropout_prob: float,
            hint_teacher_forcing: float,
            hint_repred_mode='soft',
            # nb_dims=None,
            nb_msg_passing_steps=1,
            name: str = 'dfa_net',
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
                         nb_dims=None,
                         nb_msg_passing_steps=nb_msg_passing_steps,
                         name=name)

    def __call__(self, features_list: List[_Features],
                 repred: bool,
                 algorithm_index: int,
                 return_hints: bool,
                 return_all_outputs: bool):
        """Process one batch of data.

        Args:
          features_list: A list of _Features objects, each with the inputs, hints
            and lengths for a batch of data corresponding to one algorithm.
            The list should have either length 1, at train/evaluation time,
            or length equal to the number of algorithms this Net is meant to
            process, at initialization.
          repred: False during training, when we have access to ground-truth hints.
            True in validation/test mode, when we have to use our own
            hint predictions.
          algorithm_index: Which algorithm is being processed. It can be -1 at
            initialization (either because we are initialising the parameters of
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
            padded_edge_indices_dict = features.padded_edge_indices_dict
            hint_len = features.mask_dict['hint_len']

            batch_size, nb_nodes_padded, nb_gkt_edges_padded = _dfa_data_dimensions(features)

            # YZDTODO 这么看不知道trace_i是不是没有存在的必要了？
            nb_mp_steps = max(1, trace_h.data.shape[0] - 1)
            hiddens = jnp.zeros((batch_size, nb_nodes_padded, self.hidden_dim))
            print(f'dfa_nets line 134, nb_mp_steps = {nb_mp_steps}')
            if self.use_lstm:
                lstm_state = lstm_init(batch_size * nb_nodes_padded)
                lstm_state = jax.tree_util.tree_map(
                    lambda x, b=batch_size, n=nb_nodes_padded: jnp.reshape(x, [b, n, -1]),
                    lstm_state)
            #     YZDTODO 这个不太理解，但是lstm_state是个...function?
            else:
                lstm_state = None

            mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
                pred_trace_h_i=None,
                pred_trace_o=None,
                hiddens=hiddens,
                lstm_state=lstm_state)

            # Do the first step outside of the scan because it has a different
            # computation graph.
            common_args = dict(
                input_dp_list=input_dp_list,
                trace_h=trace_h,
                hint_len=hint_len,
                batch_size=batch_size,
                nb_nodes_padded=nb_nodes_padded,
                nb_gkt_edges_padded=nb_gkt_edges_padded,
                padded_edge_indices_dict=padded_edge_indices_dict,
                encs=self.encoders[algorithm_index],
                decs=self.decoders[algorithm_index],
                repred=repred,
                return_hints=return_hints,
                return_all_outputs=return_all_outputs
            )
            mp_state, lean_mp_state = self._dfa_msg_passing_step(
                mp_state=mp_state,
                i=0,
                first_step=True,
                **common_args
            )
            if return_hints and return_all_outputs:
                print(  # [2(B), 750]; [2(B), 750]
                    f'dfa_nets line 173, mp_state.pred_trace_o:{mp_state.pred_trace_o.shape}')
                print(f'lean_mp_state.pred_trace_o: {lean_mp_state.pred_trace_o.shape}')
                print(  # [2(B), 750]; [2(B), 750]
                    f'dfa_nets line 177, mp_state.pred_trace_h_i:{mp_state.pred_trace_h_i.shape}')
                print(f'lean_mp_state.pred_trace_h_i: {lean_mp_state.pred_trace_h_i.shape}')
            # Then scan through the rest.
            scan_fn = functools.partial(
                self._dfa_msg_passing_step,
                first_step=False,
                **common_args)

            output_mp_state, accum_mp_state = hk.scan(
                scan_fn,
                mp_state,
                jnp.arange(nb_mp_steps - 1) + 1,
                length=nb_mp_steps - 1)
        if return_hints and return_all_outputs:
            print(  # [2, 750]; [3(T), 2, 750]
                f'dfa_nets line 191,output_mp_state.pred_trace_o: {output_mp_state.pred_trace_o.shape}; accum_mp_state.pred_trace_o:{accum_mp_state.pred_trace_o.shape}')
            print(  # [2, 750]; [3(T), 2, 750]
                f'dfa_nets line 194,output_mp_state.pred_trace_h_i: {output_mp_state.pred_trace_h_i.shape}; accum_mp_state.pred_trace_h_i:{accum_mp_state.pred_trace_h_i.shape}')
        # We only return the last algorithm's output. That's because
        # the output only matters when a single algorithm is processed; the case
        # `algorithm_index==-1` (meaning all algorithms should be processed)
        # is used only to init parameters.
        accum_mp_state = jax.tree_util.tree_map(
            lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
            lean_mp_state, accum_mp_state)
        if return_hints and return_all_outputs:
            print(
                f'dfa_nets line 204, accum_mp_state.pred_trace_o: {accum_mp_state.pred_trace_o.shape}')
            print(f'accumu_mp_state.pred_trace_h_i: {accum_mp_state.pred_trace_h_i.shape}')

        def invert(d):
            """Dict of lists -> list of dicts."""
            if d:
                return [dict(zip(d, i)) for i in zip(*d.values())]

        if return_all_outputs:
            # output_preds = {k: jnp.stack(v)
            #                 for k, v in accum_mp_state.output_preds.items()}
            pred_trace_o = jnp.stack(accum_mp_state.pred_trace_o)
            print(f'dfa_nets line 205, accum_mp_state.pred_trace_o: {accum_mp_state.pred_trace_o.shape}')
        #     YZDTODO 所以这个stack有啥用呢？？哦哦...可能是把所有的hints给stack到一起...
        else:
            pred_trace_o = output_mp_state.pred_trace_o
        # hint_preds = invert(accum_mp_state.hint_preds)
        pred_trace_h_i = accum_mp_state.pred_trace_h_i
        return pred_trace_o, pred_trace_h_i

    def _dfa_msg_passing_step(self,
                              mp_state: _MessagePassingScanState,
                              i: int,
                              input_dp_list: _Trajectory,
                              trace_h: probing.DataPoint,
                              hint_len: _chex_Array,
                              batch_size: int,
                              nb_nodes_padded: int,
                              nb_gkt_edges_padded: int,
                              padded_edge_indices_dict: Dict[str, _chex_Array],
                              encs: Dict[str, List[hk.Module]],
                              decs: Dict[str, Tuple[hk.Module]],
                              repred: bool,
                              first_step: bool,
                              return_hints: bool,
                              return_all_outputs: bool
                              ):
        trace_h_i = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[i], trace_h)
        if self.decode_hints and not first_step:
            assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
            hard_postprocess = (self._hint_repred_mode == 'hard' or
                                (self._hint_repred_mode == 'hard_on_eval' and repred))
            if hard_postprocess:
                decoded_trace_h_i = probing.DataPoint(name='trace_h',
                                                      location=specs.Location.EDGE,
                                                      type_=specs.Type.MASK,
                                                      data=(mp_state.pred_trace_h_i > 0.0) * 1.0)
            else:
                decoded_trace_h_i = probing.DataPoint(name='trace_h',
                                                      location=specs.Location.EDGE,
                                                      type_=specs.Type.MASK,
                                                      data=jax.nn.sigmoid(mp_state.pred_trace_h_i))
            if repred:
                trace_h_i = decoded_trace_h_i
            elif self._hint_teacher_forcing < 1.0:
                force_mask = jax.random.bernoulli(
                    hk.next_rng_key(), self._hint_teacher_forcing,
                    (batch_size,))
                # force_mask = jnp.repeat(force_mask, nb_gkt_edges)
                force_masked_data = jnp.where(jnp.expand_dims(force_mask, axis=-1),
                                              trace_h_i.data,
                                              decoded_trace_h_i.data)
                trace_h_i = probing.DataPoint(name='trace_h',
                                              location=specs.Location.EDGE,
                                              type_=specs.Type.MASK,
                                              data=force_masked_data)

        hiddens, pred_trace_o_cand, hint_preds, lstm_state = self._dfa_one_step_pred(
            input_dp_list=input_dp_list,
            trace_h_i=trace_h_i,
            hidden=mp_state.hiddens,
            batch_size=batch_size,
            nb_nodes_padded=nb_nodes_padded,
            nb_gkt_edges_padded=nb_gkt_edges_padded,
            padded_edge_indices_dict=padded_edge_indices_dict,
            lstm_state=mp_state.lstm_state,
            encs=encs,
            decs=decs,
            repred=repred)
        print(
            f'dfa_nets line 272 hiddens: {hiddens.shape}; pred_trace_o_cand: {pred_trace_o_cand.shape}; hint_preds: {hint_preds.shape}, lstm_state: {type(lstm_state)}')
        if first_step:
            pred_trace_o = pred_trace_o_cand
        else:
            # output_preds = {}
            # tmp = mask_dict['hint_len']
            print(f'dfa_nets line 270-271, i = {i}; hint_len: {type(hint_len)}\nhint_len = {hint_len}')
            is_not_done = nets._is_not_done_broadcast(hint_len, i,
                                                      pred_trace_o_cand)
            pred_trace_o = is_not_done * pred_trace_o_cand + (
                    1.0 - is_not_done) * mp_state.pred_trace_o

        new_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            pred_trace_h_i=hint_preds,
            pred_trace_o=pred_trace_o,
            hiddens=hiddens,
            lstm_state=lstm_state)
        # Save memory by not stacking unnecessary fields
        accum_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            pred_trace_h_i=hint_preds if return_hints else None,
            pred_trace_o=pred_trace_o if return_all_outputs else None,
            hiddens=None, lstm_state=None)

        # Complying to jax.scan, the first returned value is the state we carry over
        # the second value is the output that will be stacked over steps.
        return new_mp_state, accum_mp_state

    def _dfa_one_step_pred(
            self,
            input_dp_list: _Trajectory,
            trace_h_i: _chex_Array,
            hidden: _chex_Array,
            batch_size: int,
            nb_nodes_padded: int,
            nb_gkt_edges_padded: int,
            padded_edge_indices_dict: Dict[str, _chex_Array],
            lstm_state: Optional[hk.LSTMState],
            encs: Dict[str, List[hk.Module]],
            decs: Dict[str, Tuple[hk.Module]],
            repred: bool,
    ):
        """Generates one-step predictions."""

        # Initialise empty node/edge/graph features and adjacency matrix.
        node_fts = jnp.zeros((batch_size, nb_nodes_padded, self.hidden_dim))
        gkt_edge_fts = jnp.zeros((batch_size, nb_gkt_edges_padded, self.hidden_dim))

        # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Encode node/edge/graph features from inputs and (optionally) hints.
        # encode node fts
        dp_list_to_encode = input_dp_list[:]
        if self.encode_hints:
            dp_list_to_encode.append(trace_h_i)

        for dp in dp_list_to_encode:
            encoder = encs[dp.name]
            gkt_edge_fts = encoders.accum_edge_fts(encoder, dp, gkt_edge_fts)
            node_fts = encoders.accum_node_fts(encoder, dp, node_fts)

        # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('dfa_nets line 333, \ncfg_indices_padded: {}; \ngkt_indices_padded: {}'.format(
            padded_edge_indices_dict['cfg_indices_padded'].shape, padded_edge_indices_dict['gkt_indices_padded'].shape))
        nxt_hidden = hidden
        for _ in range(self.nb_msg_passing_steps):
            nxt_hidden, nxt_edge = self.processor(
                node_fts=node_fts,
                gkt_edge_fts=gkt_edge_fts,
                hidden=hidden,
                cfg_indices_padded=padded_edge_indices_dict['cfg_indices_padded'],
                gkt_indices_padded=padded_edge_indices_dict['gkt_indices_padded'],
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
        pred_trace_o, pred_trace_h_i = dfa_decoders.decode_fts(
            decoders=decs,
            h_t=h_t,
            gkt_edge_fts=gkt_edge_fts,
            gkt_edge_indices=padded_edge_indices_dict['gkt_indices_padded']
        )
        return nxt_hidden, pred_trace_o, pred_trace_h_i, nxt_lstm_state


def _dfa_data_dimensions(features: _Features) -> Tuple[int, int, int]:
    """Returns (batch_size, nb_nodes)."""
    batch_size = None
    nb_nodes_padded = None
    nb_gkt_edges_padded = None
    for inp in features.input_dp_list:
        if inp.name in ['pos', 'if_pp', 'if_ip']:
            if batch_size is None:
                batch_size, nb_nodes_padded = inp.data.shape[:2]
            else:
                assert inp.data.shape[:2] == (batch_size, nb_nodes_padded)
        if inp.name in ['gen', 'kill', 'trace_i']:
            if nb_gkt_edges_padded is None:
                nb_gkt_edges_padded = inp.data.shape[1]
            else:
                assert inp.data.shape[:2] == (batch_size, nb_gkt_edges_padded)

    return batch_size, nb_nodes_padded, nb_gkt_edges_padded
