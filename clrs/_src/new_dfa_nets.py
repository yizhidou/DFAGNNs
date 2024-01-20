import functools

from typing import Dict, List, Optional, Tuple, Union

import chex

from clrs._src import dfa_decoders
from clrs._src import encoders
from clrs._src import probing
from clrs._src import new_dfa_processors, new_dfa_utils
from clrs._src import samplers, new_dfa_samplers
from clrs._src import specs
from clrs._src import nets

import haiku as hk
import jax
import jax.numpy as jnp
import abc

_chex_Array = chex.Array
# _DataPoint = probing.DataPoint
_Features = new_dfa_samplers.Features
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
            processor_factory: new_dfa_processors.DFAProcessorFactory,
            use_lstm: bool,
            encoder_init: str,
            dropout_prob: float,
            hint_teacher_forcing: float,
            # nb_msg_passing_steps = 1,
            hint_repred_mode: str,
            if_dfa=True,
            # nb_dims=None,
            name: str = 'dfa_net',
    ):
        print('dfa_nets line 56, in dfa_nets.__init__')
        """Constructs a `Net`."""
        self.if_dfa = if_dfa
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
                         nb_msg_passing_steps=1,
                         name=name)

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
        print('dfa_nets line 105, in dfa_nets.__call__')
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
            nb_mp_steps = max(1, trace_h.data.shape[0] - 1)

            batch_size, node_fts_shape_lead, nb_edges_padded = _dfa_data_dimensions(if_dfa=self.if_dfa,
                                                                                    features=features)
            # if this is the dfa algorithm, node_fts_shape_lead=(nb_nodes, nb_bits_each);
            # otherwise, node_fts_shape_lead=(nb_nodes, ).

            print(f'dfa_nets line 135, in dfa_nets.__call__, nb_mp_steps = {nb_mp_steps}')  # checked
            if self.use_lstm and self.if_dfa:
                nb_nodes, nb_bits_each = node_fts_shape_lead
                lstm_state = lstm_init(batch_size=batch_size * nb_nodes * nb_bits_each)
                lstm_state = jax.tree_util.tree_map(
                    lambda x, b=batch_size, n=nb_nodes, m=nb_bits_each: jnp.reshape(x, [b, n, m, -1]),
                    lstm_state)
                # print(
                #     f'dfa_nets line 141, in dfa_nets.__call lstm_hidden: {lstm_state.hidden.shape}; lstm_cell: {lstm_state.cell.shape}')
            #     [B, N, hidden_dim], [B, N, hidden_dim]
            elif self.use_lstm and not self.if_dfa:
                nb_nodes = node_fts_shape_lead[0]
                lstm_state = lstm_init(batch_size=batch_size * nb_nodes)
                lstm_state = jax.tree_util.tree_map(
                    lambda x, b=batch_size, n=nb_nodes: jnp.reshape(x, [b, n, -1]),
                    lstm_state)
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
            # # \begin{commit_1}
            # print('new_dfa_nets line 188, in dfa_nets.__call__, the **first** call of _dfa_msg_passing_step is done:')
            # print(
            #     f'mp_state.pred_trace_o:{mp_state.pred_trace_o.shape}; mp_state.pred_trace_h_i: {mp_state.pred_trace_h_i.shape}')
            # if return_hints:
            #     print('since return_hint is True:')
            #     print(f'lean_mp_state.pred_trace_h_i: {lean_mp_state.pred_trace_h_i.shape}')
            # else:
            #     print('return_hint is False: ')
            #     print(f'lean_mp_state.pred_trace_h_i = {lean_mp_state.pred_trace_h_i}')
            # if return_all_outputs:
            #     print('since return_all_output is True:')
            #     print(f'lean_mp_state.pred_trace_o: {lean_mp_state.pred_trace_o.shape}')
            # else:
            #     print('return_all_output is False')
            #     print(f'lean_mp_state.pred_trace_o = {lean_mp_state.pred_trace_o}')
            # # \end{commit_1}
            # # Then scan through the rest.
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
            # print('dfa_nets line 200, in dfa_nets.__call__, the **scan** call of _dfa_msg_passing_step is done:')
            # print(
            #     f'output_mp_state.pred_trace_o:{output_mp_state.pred_trace_o.shape}; output_mp_state.pred_trace_h_i: {output_mp_state.pred_trace_h_i.shape}')
            # if return_hints:
            #     print('since return_hint is True:')
            #     print(f'accum_mp_state.pred_trace_h_i: {accum_mp_state.pred_trace_h_i.shape}')
            # else:
            #     print('return_hint is False: ')
            #     print(f'accum_mp_state.pred_trace_h_i = {accum_mp_state.pred_trace_h_i}')
            # if return_all_outputs:
            #     print('since return_all_output is True:')
            #     print(f'accum_mp_state.pred_trace_o: {accum_mp_state.pred_trace_o.shape}')
            # else:
            #     print('return_all_output is False')
            #     print(f'accum_mp_state.pred_trace_o = {accum_mp_state.pred_trace_o}')
            # # \end{commit_2}
        # We only return the last algorithm's output. That's because
        # the output only matters when a single algorithm is processed; the case
        # `algorithm_index==-1` (meaning all algorithms should be processed)
        # is used only to init parameters.
        if return_hints:
            print(f'in dfa_nets line 219, in dfa_nets.__call__, before the concat of lean_mp_state and accum_mp_state:')
            print(f'lean_mp_state.pred_trace_h_i: {lean_mp_state.pred_trace_h_i.shape}')
            print(f'accum_mp_state.pred_trace_h_i: {accum_mp_state.pred_trace_h_i.shape}')
        accum_mp_state = jax.tree_util.tree_map(
            lambda init, tail: jnp.concatenate([init[None], tail], axis=0),
            lean_mp_state, accum_mp_state)
        print('dfa_nets line 221, in dfa_nets.__call__, after the concat of lean_mp_state and accum_mp_state:')
        if return_hints:
            print('since return_hint is True:')
            print(f'concated accum_mp_state.pred_trace_h_i: {accum_mp_state.pred_trace_h_i.shape}')
        else:
            print('since return_hint is False: ')
            print(f'accum_mp_state.pred_trace_h_i = {accum_mp_state.pred_trace_h_i}')
        if return_all_outputs:
            print('since return_all_output is True:')
            print(f'concated accum_mp_state.pred_trace_o: {accum_mp_state.pred_trace_o.shape}')
        else:
            print('since return_all_output is False')
            print(f'accum_mp_state.pred_trace_o = {accum_mp_state.pred_trace_o}')

        if return_all_outputs:
            pred_trace_o = jnp.stack(accum_mp_state.pred_trace_o)
            print('since return_all_output=True,')
            print(
                f'dfa_nets line 242, in dfa_nets.__call__, after stack, accum_mp_state.pred_trace_o: {accum_mp_state.pred_trace_o.shape}')
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
        print('new_dfa_nets line 267, in dfa_nets._dfa_msg_passing_step')
        print(f'i = {i}; repred = {repred}; return_hints = {return_hints}; return_all_outputs = {return_all_outputs}')
        trace_h_i = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[i], trace_h)
        if self.decode_hints and not first_step:
            assert self._hint_repred_mode in ['soft', 'hard', 'hard_on_eval']
            hard_postprocess = (self._hint_repred_mode == 'hard' or
                                (self._hint_repred_mode == 'hard_on_eval' and repred))
            if hard_postprocess:
                decoded_trace_h_i = probing.DataPoint(name='trace_h',
                                                      location=specs.Location.NODE if self.if_dfa else specs.Location.EDGE,
                                                      type_=specs.Type.MASK,
                                                      data=(mp_state.pred_trace_h_i > 0.0) * 1.0)
            else:
                decoded_trace_h_i = probing.DataPoint(name='trace_h',
                                                      location=specs.Location.NODE if self.if_dfa else specs.Location.EDGE,
                                                      type_=specs.Type.MASK,
                                                      data=jax.nn.sigmoid(mp_state.pred_trace_h_i))
            if repred:
                trace_h_i = decoded_trace_h_i
            elif self._hint_teacher_forcing < 1.0:
                force_mask = jax.random.bernoulli(
                    hk.next_rng_key(), self._hint_teacher_forcing,
                    (batch_size,))
                print(
                    f'new_dfa_nets line 308, force_mask: {force_mask.shape}; trace_h_i: {trace_h_i.data.shape}; decoded_trace_h_i: {decoded_trace_h_i.data.shape}')
                # force_mask = jnp.repeat(force_mask, nb_gkt_edges)
                force_masked_data = jnp.where(new_dfa_utils.dim_expand_to(force_mask, trace_h_i.data),
                                              trace_h_i.data,
                                              decoded_trace_h_i.data)
                print('new_dfa_nets line 314 all good')
                trace_h_i = probing.DataPoint(name='trace_h',
                                              location=specs.Location.NODE if self.if_dfa else specs.Location.EDGE,
                                              type_=specs.Type.MASK,
                                              data=force_masked_data)

        hiddens, pred_trace_o_cand, hint_preds, lstm_state = self._dfa_one_step_pred(
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
        print(
            f'dfa_nets line 311, in dfa_nets._dfa_msg_passing_step, the call of _dfa_one_step_pred (processor) is done')
        # print(f'hiddens: {hiddens.shape}; pred_trace_o_cand: {pred_trace_o_cand.shape}; hint_preds: {hint_preds.shape}, lstm_state: {type(lstm_state)}')
        if first_step:
            pred_trace_o = pred_trace_o_cand
        else:
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

    @abc.abstractmethod
    def _dfa_one_step_pred(self,
                           hidden: _chex_Array,
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


class DFANet_v1(DFANet):
    # the old version
    def _dfa_one_step_pred(self,
                           hidden: _chex_Array,
                           input_dp_list: _Trajectory,
                           trace_h_i: probing.DataPoint,
                           batch_size: int,
                           node_fts_shape_lead: Tuple[int],
                           nb_edges_padded: int,  # gkt edges
                           padded_edge_indices_dict: Dict[str, _chex_Array],  # only cfg edges
                           lstm_state: Optional[hk.LSTMState],
                           encs: Dict[str, List[hk.Module]],
                           decs: Dict[str, Tuple[hk.Module]],
                           repred: bool,
                           first_step: bool
                           ):
        nb_nodes = node_fts_shape_lead[0]
        node_fts = jnp.zeros((batch_size, nb_nodes, self.hidden_dim))
        gkt_edge_fts = jnp.zeros((batch_size, nb_edges_padded, self.hidden_dim))

        # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Encode node/edge/graph features from inputs and (optionally) hints.
        # encode node fts
        dp_list_to_encode = input_dp_list[:]
        if self.encode_hints or first_step:
            dp_list_to_encode.append(trace_h_i)

        for dp in dp_list_to_encode:
            dp_name, dp_loc = dp.name, dp.location
            encoder = encs[dp.name]
            if dp.location == specs.Location.EDGE and dp_name != 'cfg':
                gkt_edge_fts = encoders.accum_edge_fts(encoder, dp, gkt_edge_fts)
            node_fts = encoders.accum_node_fts(encoder, dp, node_fts)

        # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # print('dfa_nets line 333, \ncfg_indices_padded: {}; \ngkt_indices_padded: {}'.format(   # checked [B, E, 2]
        #     padded_edge_indices_dict['cfg_indices_padded'].shape, padded_edge_indices_dict['gkt_indices_padded'].shape))
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
            nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(inputs=nxt_hidden, prev_state=lstm_state)
        else:
            nxt_lstm_state = None

        h_t = jnp.concatenate([node_fts, hidden, nxt_hidden], axis=-1)  # [B, N, 3*hidden_dim]
        # print(f'dfa_nets line 372, h_t: {h_t.shape}')   # checked
        if nxt_edge is not None:
            e_t = jnp.concatenate([gkt_edge_fts, nxt_edge], axis=-1)
        else:
            e_t = gkt_edge_fts

        # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decode features and (optionally) hints.
        pred_trace_o, pred_trace_h_i = dfa_decoders.decode_fts(
            decoders=decs,
            h_t=h_t,
            gkt_edge_fts=e_t,
            gkt_edge_indices=padded_edge_indices_dict['gkt_indices_padded']
        )
        # print('dfa_nets line 408, in dfa_nets._dfa_one_step_pred, after prediction:')
        # print(f'pred_trace_o: {pred_trace_o.shape}; pred_trace_h_i: {pred_trace_h_i.shape}')
        return nxt_hidden, pred_trace_o, pred_trace_h_i, nxt_lstm_state


class DFANet_v2(DFANet):
    def _dfa_one_step_pred(
            self,
            hidden: _chex_Array,
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
        print(f'new_dfa_nets line 468, in dfa_nets._dfa_one_step_pred')
        # Initialise empty node/edge/graph features and adjacency matrix.
        nb_nodes, nb_bits_each = node_fts_shape_lead
        node_fts = jnp.zeros((batch_size, nb_nodes, nb_bits_each, self.hidden_dim))
        cfg_edge_fts = jnp.zeros((batch_size, nb_edges_padded, self.hidden_dim))

        # ENCODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Encode node/edge/graph features from inputs and (optionally) hints.
        dp_list_to_encode = input_dp_list[:]
        if self.encode_hints or first_step:
            dp_list_to_encode.append(trace_h_i)
        for dp in dp_list_to_encode:
            encoder = encs[dp.name]
            if dp.location == specs.Location.EDGE:
                cfg_edge_fts = encoders.accum_edge_fts(encoder, dp, cfg_edge_fts)
            if dp.location == specs.Location.NODE:
                node_fts = encoders.accum_node_fts(encoder, dp, node_fts)
        # PROCESS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # print('dfa_nets line 333, \ncfg_indices_padded: {}; \ngkt_indices_padded: {}'.format(   # checked [B, E, 2]
        #     padded_edge_indices_dict['cfg_indices_padded'].shape, padded_edge_indices_dict['gkt_indices_padded'].shape))
        nxt_hidden = self.processor(
            node_fts=node_fts,
            edge_fts=cfg_edge_fts,
            hidden=hidden,
            cfg_indices_padded=padded_edge_indices_dict['cfg_indices_padded'])
        if not repred:  # dropout only on training
            nxt_hidden = hk.dropout(hk.next_rng_key(), self._dropout_prob, nxt_hidden)

        if self.use_lstm:
            # lstm doesn't accept multiple batch dimensions (in our case, batch and
            # nodes), so we vmap over the (first) batch dimension.
            print(
                f'new_dfa_nets line 495, nxt_hidden: {nxt_hidden.shape}; lstm.hidden: {lstm_state.hidden.shape}; lstm.cell: {lstm_state.cell.shape}')
            nxt_hidden, nxt_lstm_state = jax.vmap(self.lstm)(inputs=nxt_hidden, prev_state=lstm_state)
        else:
            nxt_lstm_state = None
        # DECODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Decode features and (optionally) hints.
        trace_h_decoder = decs['trace_h'][0]
        trace_o_decoder = decs['trace_o'][0]
        pred_trace_o = jnp.squeeze(trace_h_decoder(nxt_hidden), -1)
        pred_trace_h_i = jnp.squeeze(trace_o_decoder(nxt_hidden), -1)
        # print('dfa_nets line 408, in dfa_nets._dfa_one_step_pred, after prediction:')
        # print(f'pred_trace_o: {pred_trace_o.shape}; pred_trace_h_i: {pred_trace_h_i.shape}')
        return nxt_hidden, pred_trace_o, pred_trace_h_i, nxt_lstm_state


def _dfa_data_dimensions(if_dfa,
                         features: _Features):
    """Returns (batch_size, nb_nodes)."""
    batch_size = None
    nb_nodes_padded = None
    nb_bits_each = None
    nb_edges_padded = None
    if if_dfa:
        for inp in features.input_dp_list:
            if inp.name in ['gen_vectors', 'kill_vectors', 'trace_o']:
                # print(f'new_dfa_net line 519, {inp.name}: {inp.data.shape}')
                if batch_size is None:
                    batch_size, nb_nodes_padded, nb_bits_each = inp.data.shape
                else:
                    batch_size, nb_nodes_padded, nb_bits_each = inp.data.shape
            if inp.name == 'cfg_edges':
                nb_edges_padded = inp.data.shape[1]
        return batch_size, (nb_nodes_padded, nb_bits_each), nb_edges_padded
    else:
        for inp in features.input_dp_list:
            if inp.name in ['pos', 'if_pp', 'if_ip']:
                if batch_size is None:
                    batch_size, nb_nodes_padded = inp.data.shape[:2]
                else:
                    assert inp.data.shape[:2] == (batch_size, nb_nodes_padded)
            if inp.name in ['gen', 'kill', 'trace_o']:
                if nb_edges_padded is None:
                    nb_edges_padded = inp.data.shape[1]
                else:
                    assert inp.data.shape[:2] == (batch_size, nb_edges_padded)

        return batch_size, (nb_nodes_padded,), nb_edges_padded
