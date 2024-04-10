# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""JAX implementation of CLRS baseline models."""

import functools
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import chex

from clrs._src import decoders
from clrs._src import dfa_losses
from clrs._src import model
from clrs._src import dfa_nets
from clrs._src import probing
from clrs._src import dfa_processors
from clrs._src import dfa_samplers
from clrs._src import specs
from clrs._src import baselines

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

_Array = chex.Array
_DataPoint = probing.DataPoint
_Features = dfa_samplers.Features
# _FeaturesChunked = samplers.FeaturesChunked
_Feedback = dfa_samplers.Feedback
_Location = specs.Location
_Seed = jnp.integer
_Spec = specs.Spec
_Stage = specs.Stage
_Trajectory = dfa_samplers.Trajectory
_Type = specs.Type
_OutputClass = specs.OutputClass


# pytype: disable=signature-mismatch

class DFABaselineModel(model.Model):
    """Model implementation with selectable message passing algorithm."""

    def __init__(
            self,
            processor_factory: dfa_processors.DFAProcessorFactory,
            spec: Union[_Spec, List[_Spec]],
            # dummy_trajectory: Union[List[_Feedback], _Feedback],
            hidden_dim: int,
            encode_hints: bool,
            decode_hints: bool,
            take_hint_as_outpt: bool,
            use_lstm: bool,
            dropout_prob: float,
            hint_teacher_forcing: float,
            hint_repred_mode: str,
            learning_rate: float,
            grad_clip_max_norm: float,
            checkpoint_path: str,
            freeze_processor: bool,
            version_of_DFANet: Union[None, int],
            dfa_version: Union[None, int],
            encoder_init: str = 'default',
            name: str = 'dfa_base_model'):
        """Constructor for BaselineModel.

        The model consists of encoders, processor and decoders. It can train
        and evaluate either a single algorithm or a set of algorithms; in the
        latter case, a single processor is shared among all the algorithms, while
        the encoders and decoders are separate for each algorithm.

        Args:
          spec: Either a single spec for one algorithm, or a list of specs for
            multiple algorithms to be trained and evaluated.
          dummy_trajectory: Either a single feedback batch, in the single-algorithm
            case, or a list of feedback batches, in the multi-algorithm case, that
            comply with the `spec` (or list of specs), to initialize network size.
          processor_factory: A callable that takes an `out_size` parameter
            and returns a processor (see `processors.py`).
          hidden_dim: Size of the hidden state of the model, i.e., size of the
            message-passing vectors.
          encode_hints: Whether to provide hints as model inputs.
          decode_hints: Whether to provide hints as model outputs.
          encoder_init: The initialiser type to use for the encoders.
          use_lstm: Whether to insert an LSTM after message passing.
          learning_rate: Learning rate for training.
          grad_clip_max_norm: if greater than 0, the maximum norm of the gradients.
          checkpoint_path: Path for loading/saving checkpoints.
          freeze_processor: If True, the processor weights will be frozen and
            only encoders and decoders (and, if used, the lstm) will be trained.
          dropout_prob: Dropout rate in the message-passing stage.
          hint_teacher_forcing: Probability of using ground-truth hints instead
            of predicted hints as inputs during training (only relevant if
            `encode_hints`=True)
          hint_repred_mode: How to process predicted hints when fed back as inputs.
            Only meaningful when `encode_hints` and `decode_hints` are True.
            Options are:
              - 'soft', where we use softmaxes for categoricals, pointers
                  and mask_one, and sigmoids for masks. This will allow gradients
                  to flow through hints during training.
              - 'hard', where we use argmax instead of softmax, and hard
                  thresholding of masks. No gradients will go through the hints
                  during training; even for scalar hints, which don't have any
                  kind of post-processing, gradients will be stopped.
              - 'hard_on_eval', which is soft for training and hard for evaluation.
          name: Model name.
          nb_msg_passing_steps: Number of message passing steps per hint.

        Raises:
          ValueError: if `encode_hints=True` and `decode_hints=False`.
        """
        super(DFABaselineModel, self).__init__(spec=spec)
        # if 'dfa' in self._spec[0].keys():
        #     self.dfa_version = 0
        # elif 'dfa_v1' in self._spec[0].keys():
        #     self.dfa_version = 1
        # elif 'dfa_v2' in self._spec[0].keys():
        #     self.dfa_version = 2
        # else:
        #     self.dfa_version = None

        if encode_hints and not decode_hints:
            raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')

        assert hint_repred_mode in ['soft', 'hard', 'hard_on_eval']

        self.decode_hints = decode_hints
        self.take_hint_as_outpt = take_hint_as_outpt
        self.checkpoint_path = checkpoint_path
        self.name = name
        self._freeze_processor = freeze_processor
        if grad_clip_max_norm != 0.0:
            optax_chain = [optax.clip_by_global_norm(grad_clip_max_norm),
                           optax.scale_by_adam(),
                           optax.scale(-learning_rate)]
            self.opt = optax.chain(*optax_chain)
        else:
            self.opt = optax.adam(learning_rate)

        self._create_net_fns(hidden_dim, encode_hints, processor_factory, use_lstm,
                             encoder_init, dropout_prob, hint_teacher_forcing,
                             hint_repred_mode, version_of_DFANet, dfa_version)
        self._device_params = None
        self._device_opt_state = None
        self.opt_state_skeleton = None

    def _create_net_fns(self, hidden_dim: int,
                        encode_hints: bool,
                        processor_factory: dfa_processors.DFAProcessorFactory,
                        use_lstm: bool,
                        encoder_init: str,
                        dropout_prob: float,
                        hint_teacher_forcing: float,
                        hint_repred_mode: str,
                        version_of_DFANet: int,
                        dfa_version: Union[None, int]):
        # print('dfa_baselines line 162~ in __init__._create_net_fns')

        def _use_net(features_list: List[_Features],
                     repred: bool,
                     algorithm_index: int,
                     return_hints: bool,
                     return_all_outputs: bool):
            print('dfa_baselines line 168~ in _use_net')
            print(jax.local_devices())
            # exit(666)
            if version_of_DFANet == 2:
                return dfa_nets.DFANet_v2(spec=self._spec,
                                          hidden_dim=hidden_dim,
                                          encode_hints=encode_hints,
                                          decode_hints=self.decode_hints,
                                          processor_factory=processor_factory,
                                          use_lstm=use_lstm,
                                          encoder_init=encoder_init,
                                          dropout_prob=dropout_prob,
                                          hint_teacher_forcing=hint_teacher_forcing,
                                          hint_repred_mode=hint_repred_mode,
                                          take_hint_as_outpt=self.take_hint_as_outpt,
                                          dfa_version=dfa_version)(features_list,
                                                                   repred,
                                                                   algorithm_index,
                                                                   return_hints,
                                                                   return_all_outputs)
            elif version_of_DFANet == 3:
                return dfa_nets.DFANet_v3(spec=self._spec,
                                          hidden_dim=hidden_dim,
                                          encode_hints=encode_hints,
                                          decode_hints=self.decode_hints,
                                          processor_factory=processor_factory,
                                          use_lstm=use_lstm,
                                          encoder_init=encoder_init,
                                          dropout_prob=dropout_prob,
                                          hint_teacher_forcing=hint_teacher_forcing,
                                          hint_repred_mode=hint_repred_mode,
                                          take_hint_as_outpt=self.take_hint_as_outpt,
                                          dfa_version=dfa_version)(features_list,
                                                                   repred,
                                                                   algorithm_index,
                                                                   return_hints,
                                                                   return_all_outputs)
            elif version_of_DFANet == 4 or version_of_DFANet == 6:
                return dfa_nets.DFANet_v4(spec=self._spec,
                                          hidden_dim=hidden_dim,
                                          encode_hints=encode_hints,
                                          decode_hints=self.decode_hints,
                                          processor_factory=processor_factory,
                                          use_lstm=use_lstm,
                                          encoder_init=encoder_init,
                                          dropout_prob=dropout_prob,
                                          hint_teacher_forcing=hint_teacher_forcing,
                                          hint_repred_mode=hint_repred_mode,
                                          take_hint_as_outpt=self.take_hint_as_outpt,
                                          dfa_version=dfa_version)(features_list,
                                                                   repred,
                                                                   algorithm_index,
                                                                   return_hints,
                                                                   return_all_outputs)
            elif version_of_DFANet == 5:
                return dfa_nets.DFANet_v5(spec=self._spec,
                                          hidden_dim=hidden_dim,
                                          encode_hints=encode_hints,
                                          decode_hints=self.decode_hints,
                                          processor_factory=processor_factory,
                                          use_lstm=use_lstm,
                                          encoder_init=encoder_init,
                                          dropout_prob=dropout_prob,
                                          hint_teacher_forcing=hint_teacher_forcing,
                                          hint_repred_mode=hint_repred_mode,
                                          take_hint_as_outpt=self.take_hint_as_outpt,
                                          dfa_version=dfa_version)(features_list,
                                                                   repred,
                                                                   algorithm_index,
                                                                   return_hints,
                                                                   return_all_outputs)
            elif version_of_DFANet == 7:
                return dfa_nets.DFANet_v7(spec=self._spec,
                                          hidden_dim=hidden_dim,
                                          encode_hints=encode_hints,
                                          decode_hints=self.decode_hints,
                                          processor_factory=processor_factory,
                                          use_lstm=use_lstm,
                                          encoder_init=encoder_init,
                                          dropout_prob=dropout_prob,
                                          hint_teacher_forcing=hint_teacher_forcing,
                                          hint_repred_mode=hint_repred_mode,
                                          take_hint_as_outpt=self.take_hint_as_outpt,
                                          dfa_version=dfa_version)(features_list,
                                                                   repred,
                                                                   algorithm_index,
                                                                   return_hints,
                                                                   return_all_outputs)
            elif version_of_DFANet == 8:
                return dfa_nets.DFANet_v8(spec=self._spec,
                                          hidden_dim=hidden_dim,
                                          encode_hints=encode_hints,
                                          decode_hints=self.decode_hints,
                                          processor_factory=processor_factory,
                                          use_lstm=use_lstm,
                                          encoder_init=encoder_init,
                                          dropout_prob=dropout_prob,
                                          hint_teacher_forcing=hint_teacher_forcing,
                                          hint_repred_mode=hint_repred_mode,
                                          take_hint_as_outpt=self.take_hint_as_outpt,
                                          dfa_version=dfa_version)(features_list,
                                                                   repred,
                                                                   algorithm_index,
                                                                   return_hints,
                                                                   return_all_outputs)

            assert False

        # print('dfa_baselines line 186~')
        self.net_fn = hk.transform(_use_net)
        pmap_args = dict(axis_name='batch', devices=jax.local_devices())
        n_devices = jax.local_device_count()
        func, static_arg, extra_args = (
            (jax.jit, 'static_argnums', {}) if n_devices == 1 else
            (jax.pmap, 'static_broadcasted_argnums', pmap_args))
        pmean = functools.partial(jax.lax.pmean, axis_name='batch')
        self._maybe_pmean = pmean if n_devices > 1 else lambda x: x
        extra_args[static_arg] = 3
        # self.jitted_loss = func(self._compute_loss, **extra_args)
        self.jitted_grad = func(self._compute_grad, **extra_args)
        extra_args[static_arg] = 4
        self.jitted_feedback = func(self._feedback, donate_argnums=[0, 3],
                                    **extra_args)
        extra_args[static_arg] = [3, 4, 5]
        self.jitted_predict = func(self._predict, **extra_args)
        self.jitted_soft_max_pred = func(self._get_pred_softmax_for_debug, **extra_args)
        extra_args[static_arg] = [3, 4]
        self.jitted_accum_opt_update = func(baselines.accum_opt_update, donate_argnums=[0, 2],
                                            **extra_args)

    def init(self, features: Union[_Features, List[_Features]],
             seed: _Seed):
        if not isinstance(features, list):
            assert len(self._spec) == 1
            features = [features]
        # print('dfa_baselines line 206~ in init')
        self.params = self.net_fn.init(rng=jax.random.PRNGKey(seed),
                                       features_list=features,
                                       repred=True,  # pytype: disable=wrong-arg-types  # jax-ndarray
                                       algorithm_index=-1,
                                       return_hints=False,  # supposed to be false
                                       return_all_outputs=False)
        self.opt_state = self.opt.init(self.params)
        # We will use the optimizer state skeleton for traversal when we
        # want to avoid updating the state of params of untrained algorithms.
        self.opt_state_skeleton = self.opt.init(jnp.zeros(1))

    @property
    def params(self):
        if self._device_params is None:
            return None
        return jax.device_get(baselines._maybe_pick_first_pmapped(self._device_params))

    @params.setter
    def params(self, params):
        self._device_params = baselines._maybe_put_replicated(params)

    @property
    def opt_state(self):
        # print('dfa_baselines line 236~ in property opt_state')
        if self._device_opt_state is None:
            return None
        return jax.device_get(baselines._maybe_pick_first_pmapped(self._device_opt_state))

    @opt_state.setter
    def opt_state(self, opt_state):
        self._device_opt_state = baselines._maybe_put_replicated(opt_state)

    def _loss(self, params, rng_key, feedback, algorithm_index):
        """Calculates model loss f(feedback; params)."""
        # output_preds, hint_preds \
        # print('dfa_baselines line 359~ in _loss')
        pred_trace_o, pred_trace_h_i = self.net_fn.apply(
            params, rng_key, [feedback.features],
            repred=False,
            algorithm_index=algorithm_index,
            return_hints=True,
            return_all_outputs=False)

        # nb_nodes = _nb_nodes(feedback, is_chunked=False)
        hint_len = feedback.features.mask_dict['hint_len']
        total_loss = 0.0

        # Calculate output loss.
        truth_trace_o = feedback.trace_o
        total_loss += dfa_losses.trace_o_loss(truth=truth_trace_o,
                                              pred=pred_trace_o)

        # Optionally accumulate hint losses.
        if self.decode_hints:
            truth_trace_h = feedback.features.trace_h
            total_loss += dfa_losses.trace_h_loss(truth=truth_trace_h,
                                                  preds=pred_trace_h_i,
                                                  lengths=hint_len,
                                                  take_hint_as_outpt=self.take_hint_as_outpt)
        return total_loss

    def compute_grad(
            self,
            rng_key: hk.PRNGSequence,
            feedback: _Feedback,
            algorithm_index: Optional[int] = None,
    ) -> Tuple[float, _Array]:
        """Compute gradients."""
        # print('dfa_baselines line 306~ in compute_grad')
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        assert algorithm_index >= 0

        # Calculate gradients.
        rng_keys = baselines._maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
        feedback = baselines._maybe_pmap_data(feedback)
        loss, grads = self.jitted_grad(
            self._device_params, rng_keys, feedback, algorithm_index)
        loss = baselines._maybe_pick_first_pmapped(loss)
        grads = baselines._maybe_pick_first_pmapped(grads)

        return loss, grads

    def _update_params(self, params, grads, opt_state, algorithm_index):
        # print('dfa_baselines line 385~ in _update_params')
        updates, opt_state = baselines.filter_null_grads(
            grads, self.opt, opt_state, self.opt_state_skeleton, algorithm_index)
        if self._freeze_processor:
            params_subset = baselines._filter_out_processor(params)
            updates_subset = baselines._filter_out_processor(updates)
            assert len(params) > len(params_subset)
            assert params_subset
            new_params = optax.apply_updates(params_subset, updates_subset)
            new_params = hk.data_structures.merge(params, new_params)
        else:
            new_params = optax.apply_updates(params, updates)

        return new_params, opt_state

    def _feedback(self, params, rng_key, feedback, opt_state, algorithm_index):
        # print('dfa_baselines line 252~ in _feedback')
        lss, grads = jax.value_and_grad(self._loss)(
            params, rng_key, feedback, algorithm_index)
        grads = self._maybe_pmean(grads)
        # print(f'dfa_baseline line 386 grads = {grads}')
        params, opt_state = self._update_params(params, grads, opt_state,
                                                algorithm_index)
        lss = self._maybe_pmean(lss)
        return lss, params, opt_state

    def feedback(self, rng_key: hk.PRNGSequence,
                 feedback: _Feedback,
                 algorithm_index=None) -> float:
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        # Calculate and apply gradients.
        # print('dfa_baselines line 329~ in feedback')
        rng_keys = baselines._maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
        feedback = baselines._maybe_pmap_data(feedback)
        loss, self._device_params, self._device_opt_state = self.jitted_feedback(
            self._device_params, rng_keys, feedback,
            self._device_opt_state, algorithm_index)
        loss = baselines._maybe_pick_first_pmapped(loss)
        return loss

    def _compute_grad(self, params, rng_key, feedback, algorithm_index):
        # print('dfa_baselines line 246~ in _compute_grad')
        lss, grads = jax.value_and_grad(self._loss)(
            params, rng_key, feedback, algorithm_index)
        return self._maybe_pmean(lss), self._maybe_pmean(grads)

    def _get_pred_softmax_for_debug(self, params, rng_key: hk.PRNGSequence, features: _Features,
                                    algorithm_index: int, return_hints: bool,
                                    return_all_outputs: bool):
        outs, hint_preds = self.net_fn.apply(
            params, rng_key, [features],
            repred=True, algorithm_index=algorithm_index,
            return_hints=return_hints,
            return_all_outputs=return_all_outputs)
        return outs, jax.nn.log_softmax(outs)

    def get_softmax_for_debug(self, rng_key: hk.PRNGSequence,
                              features: _Features,
                              algorithm_index: Optional[int] = None,
                              return_hints: bool = False,
                              return_all_outputs: bool = False):
        """Model inference step."""
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        # print('dfa_baselines line 346~ in predict')
        rng_keys = baselines._maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
        features = baselines._maybe_pmap_data(features)
        return baselines._maybe_restack_from_pmap(
            self.jitted_soft_max_pred(
                self._device_params, rng_keys, features,
                algorithm_index,
                return_hints,
                return_all_outputs))

    def _predict(self, params, rng_key: hk.PRNGSequence, features: _Features,
                 algorithm_index: int, return_hints: bool,
                 return_all_outputs: bool):
        # print('dfa_baselines line 264~ in _predict')
        outs, hint_preds = self.net_fn.apply(
            params, rng_key, [features],
            repred=True, algorithm_index=algorithm_index,
            return_hints=return_hints,
            return_all_outputs=return_all_outputs)
        # outs = decoders.postprocess(self._spec[algorithm_index],
        #                             outs,
        #                             sinkhorn_temperature=0.1,
        #                             sinkhorn_steps=50,
        #                             hard=True,
        #                             )
        if self._spec[algorithm_index]['trace_o'][-1] == specs.Type.MASK:
            outs = (outs > 0.0) * 1.0
            if return_hints:
                # assert hint_preds is not None
                hint_preds = (hint_preds > 0.0) * 1.0
        else:
            outs = jnp.argmax(outs, -1)
            if return_hints:
                # assert hint_preds is not None
                hint_preds = jnp.argmax(hint_preds, -1)
        return outs, hint_preds

    def predict(self, rng_key: hk.PRNGSequence, features: _Features,
                algorithm_index: Optional[int] = None,
                return_hints: bool = False,
                return_all_outputs: bool = False):
        """Model inference step."""
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        # print('dfa_baselines line 346~ in predict')
        rng_keys = baselines._maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
        features = baselines._maybe_pmap_data(features)
        return baselines._maybe_restack_from_pmap(
            self.jitted_predict(
                self._device_params, rng_keys, features,
                algorithm_index,
                return_hints,
                return_all_outputs))

    def _calculate_measures(self,
                            type: str,
                            mask,
                            truth_data,
                            pred_data):
        if type == specs.Type.MASK:
            # Convert predictions to binary by thresholding
            preds_masked = pred_data[mask]
            truth_data_masked = truth_data[mask]

            # Calculate true positives, false positives, and false negatives
            tp = jnp.sum(jnp.where(preds_masked > 0.5, 1, 0) * jnp.where(truth_data_masked > 0.5, 1, 0))
            fp = jnp.sum(jnp.where(preds_masked > 0.5, 1, 0) * jnp.where(truth_data_masked < 0.5, 1, 0))
            fn = jnp.sum(jnp.where(preds_masked < 0.5, 1, 0) * jnp.where(truth_data_masked > 0.5, 1, 0))
        else:
            preds_masked = pred_data[mask]
            truth_data_masked = jnp.argmax(truth_data[mask], -1)
            # print(f'dfa_baseline line 482, preds = {preds_masked}; truth = {truth_data_masked}')

            tp = jnp.sum(preds_masked * truth_data_masked)
            fp = jnp.sum(preds_masked * (1 - truth_data_masked))
            fn = jnp.sum((1 - preds_masked) * truth_data_masked)
        # print(f'dfa_baseline line 489, tp = {tp}; fp = {fp}; fn = {fn}')
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            # precision = jnp.array([1.])
            precision = 1.0
        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            # recall = jnp.array([1.])
            recall = 1.0
        if precision + recall > 0.0:
            f_1 = 2.0 * precision * recall / (precision + recall)
        else:
            # f_1 = jnp.array([0.])
            f_1 = 0.0
        # positive_num = jnp.sum(truth_data_masked)
        # total_num = 1
        # for i in truth_data_masked.shape:
        #     total_num *= i

        return precision, recall, f_1

    def get_measures(self, rng_key: hk.PRNGSequence,
                     feedback: _Feedback,
                     algorithm_index: Optional[int] = None,
                     return_hints: bool = False,
                     return_all_outputs: bool = False
                     ):
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        rng_keys = baselines._maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
        features = baselines._maybe_pmap_data(feedback.features)
        trace_o_pred, trace_h_pred = self.jitted_predict(self._device_params, rng_keys, features,
                                                         algorithm_index,
                                                         return_hints,
                                                         return_all_outputs)
        type_ = feedback.trace_o._type_
        if type_ == specs.Type.MASK:
            mask = feedback.trace_o.data != specs.OutputClass.MASKED
            # truth_trace_o = feedback.trace_o.data
        else:
            # assert type_ == specs.Type.CATEGORICAL
            mask = feedback.trace_o.data[..., 0] != specs.OutputClass.MASKED
            # truth_trace_o = jnp.argmax(feedback.trace_o.data, -1)
        if trace_h_pred is not None:
            # print(f'dfa_baselines line 520, return_hints = {return_hints};')
            # print(
            #     f'the shape of trace_h_pred is {trace_h_pred.shape};\nthe shape of truth.trace_h is: {jnp.argmax(feedback.features.trace_h.data[1:, ...], -1).shape}')
            # print(f'the shape of trace_o is: {trace_o_pred.shape}')
            trace_h_f1_list = []
            for time_step in range(trace_h_pred.shape[0]-1):
                pred_trace_h_i = trace_h_pred[time_step]
                if type_ == specs.Type.CATEGORICAL:
                    truth_trace_h_i = feedback.features.trace_h.data[time_step+1]
                else:
                    truth_trace_h_i = feedback.features.trace_h.data[time_step+1]
                precision_of_this_step, recall_of_this_step, f_1_of_this_step = self._calculate_measures(type=type_,
                                                                                                         mask=mask,
                                                                                                         truth_data=truth_trace_h_i,
                                                                                                         pred_data=pred_trace_h_i)
                trace_h_f1_list.append(f_1_of_this_step.item())
            print('dfa_baselines line 586, trace_h_f1_list is:')
            print(trace_h_f1_list)

        # truth_trace_o = jnp.argmax(feedback.trace_o.data[mask], -1) if type_ == specs.Type.CATEGORICAL else feedback.trace_o.data[mask]
        return self._calculate_measures(type=type_, mask=mask, truth_data=feedback.trace_o.data, pred_data=trace_o_pred)
        # if feedback.trace_o._type_ == specs.Type.MASK:
        #     # Mask out the -1 values in truth
        #     mask = feedback.trace_o.data != specs.OutputClass.MASKED
        #     # Convert predictions to binary by thresholding
        #     preds_masked = trace_o_pred[mask]
        #     truth_data_masked = feedback.trace_o.data[mask]
        #
        #     # Calculate true positives, false positives, and false negatives
        #     tp = jnp.sum(jnp.where(preds_masked > 0.5, 1, 0) * jnp.where(truth_data_masked > 0.5, 1, 0))
        #     fp = jnp.sum(jnp.where(preds_masked > 0.5, 1, 0) * jnp.where(truth_data_masked < 0.5, 1, 0))
        #     fn = jnp.sum(jnp.where(preds_masked < 0.5, 1, 0) * jnp.where(truth_data_masked > 0.5, 1, 0))
        # else:
        #     mask = feedback.trace_o.data[..., 0] != specs.OutputClass.MASKED
        #     preds_masked = trace_o_pred[mask]
        #     truth_data_masked = jnp.argmax(feedback.trace_o.data[mask], -1)
        #     # print(f'dfa_baseline line 482, preds = {preds_masked}; truth = {truth_data_masked}')
        #
        #     tp = jnp.sum(preds_masked * truth_data_masked)
        #     fp = jnp.sum(preds_masked * (1 - truth_data_masked))
        #     fn = jnp.sum((1 - preds_masked) * truth_data_masked)
        #
        # print(f'dfa_baseline line 489, tp = {tp}; fp = {fp}; fn = {fn}')
        # if tp + fp > 0:
        #     precision = tp / (tp + fp)
        # else:
        #     # precision = jnp.array([1.])
        #     precision = 1.0
        # if tp + fn > 0:
        #     recall = tp / (tp + fn)
        # else:
        #     # recall = jnp.array([1.])
        #     recall = 1.0
        # if precision + recall > 0.0:
        #     f_1 = 2.0 * precision * recall / (precision + recall)
        # else:
        #     # f_1 = jnp.array([0.])
        #     f_1 = 0.0
        # positive_num = jnp.sum(truth_data_masked)
        # total_num = 1
        # for i in truth_data_masked.shape:
        #     total_num *= i
        #
        # return precision, recall, f_1, positive_num, total_num

    def restore_model(self, file_name: str, only_load_processor: bool = False):
        """Restore model from `file_name`."""
        # print('dfa_baselines line 433~ in restore_model')
        path = os.path.join(self.checkpoint_path, file_name)
        with open(path, 'rb') as f:
            restored_state = pickle.load(f)
            if only_load_processor:
                restored_params = baselines._filter_in_processor(restored_state['params'])
            else:
                restored_params = restored_state['params']
            self.params = hk.data_structures.merge(self.params, restored_params)
            self.opt_state = restored_state['opt_state']

    def save_model(self, file_name: str):
        # print('dfa_baselines line 445~ in save_model')
        """Save model (processor weights only) to `file_name`."""
        os.makedirs(self.checkpoint_path, exist_ok=True)
        to_save = {'params': self.params, 'opt_state': self.opt_state}
        path = os.path.join(self.checkpoint_path, file_name)
        with open(path, 'wb') as f:
            pickle.dump(to_save, f)

    def update_model_params_accum(self, grads) -> None:
        grads = baselines._maybe_put_replicated(grads)
        self._device_params, self._device_opt_state = self.jitted_accum_opt_update(
            self._device_params, grads, self._device_opt_state, self.opt,
            self._freeze_processor)

    def verbose_loss(self, feedback: _Feedback, hint_preds: _Array) -> Dict[str, _Array]:
        """Gets verbose loss information."""

        # nb_nodes = _nb_nodes(feedback, is_chunked=False)
        lengths = feedback.features.lengths
        losses_ = {}

        # Optionally accumulate hint losses.
        if self.decode_hints:
            losses_.update(dfa_losses.trace_h_loss(truth=feedback.features.trace_h,
                                                   preds=hint_preds,
                                                   lengths=lengths,
                                                   verbose=True))
            # for truth in feedback.features.hints:
            #     losses_.update(
            #         losses.hint_loss(
            #             truth=truth,
            #             preds=[x[truth.name] for x in hint_preds],
            #             lengths=lengths,
            #             nb_nodes=nb_nodes,
            #             verbose=True,
            #         ))

        return losses_

    # def compute_loss(
    #         self,
    #         rng_key: hk.PRNGSequence,
    #         feedback: _Feedback,
    #         algorithm_index: Optional[int] = None,
    # ) -> float:
    #     """Compute gradients."""
    #
    #     if algorithm_index is None:
    #         assert len(self._spec) == 1
    #         algorithm_index = 0
    #     assert algorithm_index >= 0
    #     # print('dfa_baselines line 290~ in compute_loss')
    #     # Calculate gradients.
    #     rng_keys = baselines._maybe_pmap_rng_key(rng_key)  # pytype: disable=wrong-arg-types  # numpy-scalars
    #     feedback = baselines._maybe_pmap_data(feedback)
    #     loss, _ = self.jitted_grad(
    #         self._device_params, rng_keys, feedback, algorithm_index)
    #     loss = baselines._maybe_pick_first_pmapped(loss)
    #     return loss

# @functools.partial(jax.jit, static_argnums=1)
# def _pmap_data(data: Union[_Feedback, _Features], n_devices: int):
#     """Replicate/split feedback or features for pmapping."""
#     if isinstance(data, _Feedback):
#         features = data.features
#     else:
#         features = data
#     pmap_data = features._replace(
#         input_dp_list=baselines._pmap_reshape(features.input_dp_list, n_devices),
#         trace_h=baselines._pmap_reshape(features.trace_h, n_devices, split_axis=1),
#         padded_edge_indices_dict=baselines._pmap_reshape(features.padded_edge_indices_dict, n_devices),
#         mask_dict=baselines._pmap_reshape(features.mask_dict, n_devices),
#     )
#     return pmap_data
#
#
# def _maybe_pmap_data(data: Union[_Feedback, _Features]):
#     n_devices = jax.local_device_count()
#     if n_devices == 1:
#         return data
#     return _pmap_data(data, n_devices)
