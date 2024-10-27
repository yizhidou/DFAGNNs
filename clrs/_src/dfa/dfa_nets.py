import functools

from typing import Dict, List, Optional, Tuple, Union

import chex

from clrs._src import dfa_decoders
from clrs._src import encoders, decoders
from clrs._src import probing
from clrs._src import dfa_processors, dfa_utils
from clrs._src import samplers, dfa_samplers
from clrs._src import specs
from clrs._src import nets

import haiku as hk
import jax
import jax.numpy as jnp
import abc

_chex_Array = chex.Array
# _DataPoint = probing.DataPoint
_Features = dfa_samplers.Features
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
            # nb_msg_passing_steps = 1,
            hint_repred_mode: str,
            take_hint_as_outpt: bool,
            just_one_layer: bool,
            name: str = 'dfa_net',
    ):
        # print('dfa_nets line 56, in dfa_nets.__init__')
        """Constructs a `Net`."""
        self.take_hint_as_outpt = take_hint_as_outpt
        self.just_one_layer = just_one_layer


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
                         nb_dims=[dict(trace_h=2, trace_o=2)],
                         nb_msg_passing_steps=1,
                         name=name)

    def _construct_encoders_decoders(self):
        """Constructs encoders and decoders, separate for each algorithm."""
        encoders_ = []
        decoders_ = []
        enc_algo_idx = None
        for (algo_idx, spec) in enumerate(self.spec):
            enc = {}
            dec = {}
            for name, (stage, loc, t) in spec.items():
                # if stage == _Stage.INPUT or (
                #         stage == _Stage.HINT and self.encode_hints):
                if stage == _Stage.INPUT or stage == _Stage.HINT:
                    # Build input encoders.
                    if name == specs.ALGO_IDX_INPUT_NAME:
                        if enc_algo_idx is None:
                            enc_algo_idx = [hk.Linear(self.hidden_dim,
                                                      name=f'{name}_enc_linear')]
                        enc[name] = enc_algo_idx
                    else:
                        enc[name] = encoders.construct_encoders(
                            stage, loc, t, hidden_dim=self.hidden_dim,
                            init=self.encoder_init,
                            name=f'algo_{algo_idx}_{name}')

                if stage == _Stage.OUTPUT or (
                        stage == _Stage.HINT and self.decode_hints):
                    # Build output decoders.
                    dec[name] = decoders.construct_decoders(
                        loc, t, hidden_dim=self.hidden_dim,
                        nb_dims=None if self.nb_dims is None else self.nb_dims[algo_idx][name],
                        name=f'algo_{algo_idx}_{name}')
            encoders_.append(enc)
            decoders_.append(dec)

        return encoders_, decoders_
    def __call__(self, features_list: List[_Features],
                 repred: bool,
                 algorithm_index: int,
                 return_hints: bool,
                 return_all_outputs: bool = False):
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
        # print('dfa_nets line 105, in dfa_nets.__call__')
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
            # print(f'dfa_nets line 133, trace_h: {trace_h.data.shape}')
            padded_edge_indices_dict = features.padded_edge_indices_dict
            if self.take_hint_as_outpt:
                hint_len = features.mask_dict['hint_len']
                nb_mp_steps = max(1, trace_h.data.shape[0] - 1)
                print(f'dfa_nets line 190, len of trace_h is: {trace_h.data.shape[0]}; hint_len = {hint_len}; nb_mp_steps = {nb_mp_steps}')
            else:
                hint_len = features.mask_dict['hint_len'] - 1
                nb_mp_steps = max(1, trace_h.data.shape[0] - 2)

            batch_size, node_fts_shape_lead, nb_edges_padded = _dfa_data_dimensions(features=features)
            # if this is the dfa algorithm, node_fts_shape_lead=(nb_nodes, nb_bits_each);
            # otherwise, node_fts_shape_lead=(nb_nodes, ).

            # print(f'dfa_nets line 135, in dfa_nets.__call__, nb_mp_steps = {nb_mp_steps}')  # checked
            if self.use_lstm:
                nb_nodes, nb_bits_each = node_fts_shape_lead
                lstm_state = lstm_init(batch_size=batch_size * nb_nodes * nb_bits_each)
                lstm_state = jax.tree_util.tree_map(
                    lambda x, b=batch_size, n=nb_nodes, m=nb_bits_each: jnp.reshape(x, [b, n, m, -1]),
                    lstm_state)
                # print(
                #     f'dfa_nets line 141, in dfa_nets.__call lstm_hidden: {lstm_state.hidden.shape}; lstm_cell: {lstm_state.cell.shape}')
            #     [B, N, hidden_dim], [B, N, hidden_dim]
            else:
                lstm_state = None
            hiddens = jnp.zeros((batch_size,) + node_fts_shape_lead + (self.hidden_dim,))
            mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
                pred_trace_h_i=None,
                pred_trace_o=None,
                hiddens=hiddens,
                lstm_state=lstm_state)

            # Do the first step outside of the scan because it has a different
            # computation graph.
            common_args = dict(
                hint_len=hint_len,
                input_dp_list=input_dp_list,
                trace_h=trace_h,
                batch_size=batch_size,
                node_fts_shape_lead=node_fts_shape_lead,
                nb_edges_padded=nb_edges_padded,
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
            print('dfa_nets line 247, after the first layer of GNN,')
            print(f'mp_state.pred_trace_o: {mp_state.pred_trace_o.shape}; mp_state.pred_trace_h: {mp_state.pred_trace_h_i.shape}')
            if return_hints:
                print(f'lean_mp_state.pred_trace_h: {lean_mp_state.pred_trace_h_i.shape}')
            if not self.just_one_layer:
                scan_fn = functools.partial(
                    self._dfa_msg_passing_step,
                    first_step=False,
                    **common_args)

                output_mp_state, accum_mp_state = hk.scan(
                    scan_fn,
                    mp_state,
                    jnp.arange(nb_mp_steps - 1) + 1,
                    length=nb_mp_steps - 1)
                # # \begin{commit_2}
                print('dfa_nets line 263, after the scan but before concat')
                print(f'output_mp_state.pred_trace_o: {output_mp_state.pred_trace_o.shape}; output_mp_state.pred_trace_h_i: {output_mp_state.pred_trace_h_i.shape}')
                if return_hints:
                    print(f'accum_mp_state.pred_trace_h_i.shape = {accum_mp_state.pred_trace_h_i.shape}')
        # We only return the last algorithm's output. That's because
        # the output only matters when a single algorithm is processed; the case
        # `algorithm_index==-1` (meaning all algorithms should be processed)
        # is used only to init parameters.
        if self.just_one_layer:
            # assert hint_len == 1
            return mp_state.pred_trace_o, mp_state.pred_trace_h_i
        accum_mp_state = jax.tree_util.tree_map(
            lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
            lean_mp_state, accum_mp_state)
        if return_hints:
            print('dfa_nets line 278, after concat results from all layers')
            print(f'accum_mp_state.pred_trace_h_i: {accum_mp_state.pred_trace_h_i.shape}')
        if return_all_outputs:
            pred_trace_o = jnp.stack(accum_mp_state.pred_trace_o)
            # print('since return_all_output=True,')
            # print(
            #     f'dfa_nets line 242, in dfa_nets.__call__, after stack, accum_mp_state.pred_trace_o: {accum_mp_state.pred_trace_o.shape}')
        else:
            pred_trace_o = output_mp_state.pred_trace_o
        # hint_preds = invert(accum_mp_state.hint_preds)
        pred_trace_h_i = accum_mp_state.pred_trace_h_i
        return pred_trace_o, pred_trace_h_i

    def _dfa_msg_passing_step(self,
                              mp_state: _MessagePassingScanState,
                              i: int,
                              hint_len: _chex_Array,
                              input_dp_list: _Trajectory,
                              trace_h: probing.DataPoint,
                              batch_size: int,
                              node_fts_shape_lead: Tuple[int],
                              nb_edges_padded: int,
                              padded_edge_indices_dict: Dict[str, _chex_Array],
                              encs: Dict[str, List[hk.Module]],
                              decs: Dict[str, Tuple[hk.Module]],
                              repred: bool,
                              first_step: bool,
                              return_hints: bool,
                              return_all_outputs: bool
                              ):
        # print('dfa_nets line 267, in dfa_nets._dfa_msg_passing_step')
        # print(f'i = {i}; repred = {repred}; return_hints = {return_hints}; return_all_outputs = {return_all_outputs}')
        trace_h_i = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[i], trace_h)
        if first_step:
            print(f'dfa_nets line 311, this is the first GNN layer. i = {i}')
        if self.decode_hints and not first_step:
            print(f'dfa_nets line 313, this is a GNN layer but not the first layer. i = {i}')
            # exit(666)
            assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
            hard_postprocess = (self._hint_repred_mode == 'hard' or
                                (self._hint_repred_mode == 'hard_on_eval' and repred))
            if hard_postprocess:
                type_of_trace_h_i = specs.Type.CATEGORICAL
                post_processed_data_hard =hk.one_hot(jnp.argmax(mp_state.pred_trace_h_i, -1), 2)
                decoded_trace_h_i = probing.DataPoint(name='trace_h',
                                                      location=specs.Location.NODE,
                                                      type_=type_of_trace_h_i,
                                                      data=post_processed_data_hard)
            else:
                type_of_trace_h_i = specs.Type.CATEGORICAL
                post_processed_data = jax.nn.softmax(mp_state.pred_trace_h_i, axis=-1)
                decoded_trace_h_i = probing.DataPoint(name='trace_h',
                                                      location=specs.Location.NODE,
                                                      type_=type_of_trace_h_i,
                                                      data=post_processed_data)
            if repred or self._hint_teacher_forcing is None:
                trace_h_i = decoded_trace_h_i
            elif self._hint_teacher_forcing < 1.0:
                force_mask = jax.random.bernoulli(
                    hk.next_rng_key(), self._hint_teacher_forcing,
                    (batch_size,))
                # print(
                #     f'dfa_nets line 308, force_mask: {force_mask.shape}; trace_h_i: {trace_h_i.data.shape}; decoded_trace_h_i: {decoded_trace_h_i.data.shape}')
                # force_mask = jnp.repeat(force_mask, nb_gkt_edges)
                force_masked_data = jnp.where(dfa_utils.dim_expand_to(force_mask, trace_h_i.data),
                                              trace_h_i.data,
                                              decoded_trace_h_i.data)
                # print('dfa_nets line 314 all good')
                trace_h_i = probing.DataPoint(name='trace_h',
                                              location=specs.Location.NODE,
                                              type_=specs.Type.CATEGORICAL,
                                              data=force_masked_data)

        hiddens, pred_trace_o, pred_trace_i, lstm_state = self._dfa_one_step_pred(
            hidden=mp_state.hiddens,
            input_dp_list=input_dp_list,
            trace_h_i=trace_h_i,
            batch_size=batch_size,
            node_fts_shape_lead=node_fts_shape_lead,
            nb_edges_padded=nb_edges_padded,
            padded_edge_indices_dict=padded_edge_indices_dict,
            lstm_state=mp_state.lstm_state,
            encs=encs,
            decs=decs,
            repred=repred,
            first_step=first_step)
        # print(
        #     f'dfa_nets line 311, in dfa_nets._dfa_msg_passing_step, the call of _dfa_one_step_pred (processor) is done')
        # print(f'hiddens: {hiddens.shape}; pred_trace_o_cand: {pred_trace_o_cand.shape}; hint_preds: {hint_preds.shape}, lstm_state: {type(lstm_state)}')
        if first_step:
            pred_trace_o = pred_trace_o
        else:
            is_not_done = nets._is_not_done_broadcast(hint_len, i,
                                                      pred_trace_o)
            pred_trace_o = is_not_done * pred_trace_o + (
                    1.0 - is_not_done) * mp_state.pred_trace_o

        new_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            pred_trace_h_i=pred_trace_i,
            pred_trace_o=pred_trace_o,
            hiddens=hiddens,
            lstm_state=lstm_state)
        # Save memory by not stacking unnecessary fields
        accum_mp_state = _MessagePassingScanState(  # pytype: disable=wrong-arg-types  # numpy-scalars
            pred_trace_h_i=pred_trace_i if return_hints else None,
            pred_trace_o=pred_trace_o if return_all_outputs else None,
            hiddens=None, lstm_state=None)

        # Complying to jax.scan, the first returned value is the state we carry over
        # the second value is the output that will be stacked over steps.
        return new_mp_state, accum_mp_state

    @abc.abstractmethod
    def _dfa_one_step_pred(self,
                           hidden: Optional[_chex_Array],
                           input_dp_list: _Trajectory,
                           trace_h_i: probing.DataPoint,
                           batch_size: int,
                           node_fts_shape_lead: Tuple[int],
                           nb_edges_padded: int,
                           padded_edge_indices_dict: Dict[str, _chex_Array],  # only cfg edges
                           lstm_state: Optional[hk.LSTMState],
                           encs: Dict[str, List[hk.Module]],
                           decs: Dict[str, Tuple[hk.Module]],
                           repred: bool,
                           first_step: bool
                           ):
        pass

class DFANet_plus(DFANet):
    def _dfa_one_step_pred(
            self,
            hidden: Optional[_chex_Array],  # not used in this version
            input_dp_list: _Trajectory,
            trace_h_i: probing.DataPoint,
            batch_size: int,
            node_fts_shape_lead: Tuple[int],
            nb_edges_padded: int,  # cfg edges
            padded_edge_indices_dict: Dict[str, _chex_Array],  # only cfg edges
            lstm_state: Optional[hk.LSTMState],
            encs: Dict[str, List[hk.Module]],
            decs: Dict[str, Tuple[hk.Module]],
            repred: bool,
            first_step: bool
    ):
        """Generates one-step predictions."""
        nb_nodes, nb_bits_each = node_fts_shape_lead
        node_fts = jnp.zeros((batch_size, nb_nodes, nb_bits_each, self.hidden_dim))
        cfg_edge_fts = jnp.zeros((batch_size, nb_edges_padded, self.hidden_dim))

        # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Encode node/edge/graph features from inputs and (optionally) hints.
        dp_list_to_encode = input_dp_list[:]
        for dp in dp_list_to_encode:
            encoder = encs[dp.name]
            if dp.location == specs.Location.EDGE:
                cfg_edge_fts = encoders.accum_edge_fts(encoder, dp, cfg_edge_fts)
            if dp.location == specs.Location.NODE:
                node_fts = encoders.accum_node_fts(encoder, dp, node_fts)

        new_hint_state = self.processor(
            node_fts=node_fts,
            edge_fts=cfg_edge_fts,
            hint_state=encoders._encode_inputs(encoders=encs['trace_h'],
                                               dp=trace_h_i) if self.encode_hints or first_step else hidden,
            cfg_indices_padded=padded_edge_indices_dict['cfg_indices_padded'])
        if not repred:  # dropout only on training
            new_hint_state = hk.dropout(hk.next_rng_key(), self._dropout_prob, new_hint_state)

        if self.use_lstm:
            new_hint_state, nxt_lstm_state = jax.vmap(self.lstm)(inputs=new_hint_state, prev_state=lstm_state)
        else:
            nxt_lstm_state = None
        # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decode features and (optionally) hints.

        trace_o_decoder = decs['trace_o'][0]
        # print(f'dfa_nets line 640 new_hint_state: {new_hint_state.shape}')
        if self.decode_hints:
            trace_h_decoder = decs['trace_h'][0]
            pred_trace_h_i = trace_h_decoder(new_hint_state)
            if self.take_hint_as_outpt:
                pred_trace_o = pred_trace_h_i
            else:
                # pred_trace_o = jnp.squeeze(trace_o_decoder(new_hint_state), -1)
                pred_trace_o = trace_o_decoder(new_hint_state)
        else:
            pred_trace_h_i = None
            pred_trace_o = trace_o_decoder(new_hint_state)
        return new_hint_state, pred_trace_o, pred_trace_h_i, nxt_lstm_state

class DFANet_others(DFANet):
    def _dfa_one_step_pred(
            self,
            hidden: Optional[_chex_Array],  # not used in this version
            input_dp_list: _Trajectory,
            trace_h_i: probing.DataPoint,
            batch_size: int,
            node_fts_shape_lead: Tuple[int],
            nb_edges_padded: int,  # cfg edges
            padded_edge_indices_dict: Dict[str, _chex_Array],  # only cfg edges
            lstm_state: Optional[hk.LSTMState],
            encs: Dict[str, List[hk.Module]],
            decs: Dict[str, Tuple[hk.Module]],
            repred: bool,
            first_step: bool
    ):
        """Generates one-step predictions."""
        nb_nodes, nb_bits_each = node_fts_shape_lead
        node_fts = jnp.zeros((batch_size, nb_nodes, nb_bits_each, self.hidden_dim))
        cfg_edge_fts = jnp.zeros((batch_size, nb_edges_padded, self.hidden_dim))

        # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        dp_list_to_encode = input_dp_list[:]
        if self.encode_hints or first_step:
            dp_list_to_encode.append(trace_h_i)
        for dp in dp_list_to_encode:
            encoder = encs[dp.name]
            if dp.location == specs.Location.EDGE:
                cfg_edge_fts = encoders.accum_edge_fts(encoder, dp, cfg_edge_fts)
            if dp.location == specs.Location.NODE:
                print(f'dfa_nets line 871, name: {dp.name}')
                node_fts = encoders.accum_node_fts(encoder, dp, node_fts)

        nxt_hidden = self.processor(
            node_fts=node_fts,
            edge_fts=cfg_edge_fts,
            hidden=hidden,
            cfg_indices_padded=padded_edge_indices_dict['cfg_indices_padded'])
        if not repred:  # dropout only on training
            nxt_hidden = hk.dropout(hk.next_rng_key(), self._dropout_prob, nxt_hidden)

        if self.use_lstm:
            nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(inputs=nxt_hidden, prev_state=lstm_state)
        else:
            nxt_lstm_state = None
        # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decode features and (optionally) hints.

        trace_o_decoder = decs['trace_o'][0]
        if self.decode_hints:
            trace_h_decoder = decs['trace_h'][0]
            pred_trace_h_i = trace_h_decoder(nxt_hidden)
            if self.take_hint_as_outpt:
                pred_trace_o = pred_trace_h_i
            else:
                pred_trace_o = jnp.squeeze(trace_o_decoder(nxt_hidden), -1)
        else:
            pred_trace_h_i = None
            pred_trace_o = trace_o_decoder(nxt_hidden)
        # print('dfa_nets line 510, in dfa_nets._dfa_one_step_pred, after prediction:')
        # print(f'pred_trace_o: {pred_trace_o.shape}; pred_trace_h_i: {pred_trace_h_i.shape}')
        return nxt_hidden, pred_trace_o, pred_trace_h_i, nxt_lstm_state


def _dfa_data_dimensions(features: _Features):
    """Returns (batch_size, nb_nodes)."""
    batch_size = None
    nb_nodes_padded = None
    nb_bits_each = None
    nb_edges_padded = None
    for inp in features.input_dp_list:
        if inp.name in ['gen_vectors', 'kill_vectors', 'trace_o']:
            # print(f'dfa_net line 519, {inp.name}: {inp.data.shape}')
            batch_size, nb_nodes_padded, nb_bits_each = inp.data.shape[:3]
        if inp.name == 'cfg_edges':
            nb_edges_padded = inp.data.shape[1]
    return batch_size, (nb_nodes_padded, nb_bits_each), nb_edges_padded

