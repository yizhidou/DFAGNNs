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
            use_lstm: bool,
            dropout_prob: float,
            hint_teacher_forcing: float,
            hint_repred_mode: str,
            learning_rate: float,
            grad_clip_max_norm: float,
            checkpoint_path: str,
            freeze_processor: bool,
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

        if encode_hints and not decode_hints:
            raise ValueError('`encode_hints=True`, `decode_hints=False` is invalid.')

        assert hint_repred_mode in ['soft', 'hard', 'hard_on_eval']

        self.decode_hints = decode_hints
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
                             hint_repred_mode)
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
                        hint_repred_mode: str):
        # print('dfa_baselines line 162~ in __init__._create_net_fns')

        def _use_net(features_list: List[_Features],
                     repred: bool,
                     algorithm_index: int,
                     return_hints: bool,
                     return_all_outputs: bool):
            # print('dfa_baselines line 168~ in _use_net')
            # print(jax.local_devices())
            return dfa_nets.DFANet_v2(spec=self._spec,
                                      hidden_dim=hidden_dim,
                                      encode_hints=encode_hints,
                                      decode_hints=self.decode_hints,
                                      processor_factory=processor_factory,
                                      use_lstm=use_lstm,
                                      encoder_init=encoder_init,
                                      dropout_prob=dropout_prob,
                                      hint_teacher_forcing=hint_teacher_forcing,
                                      hint_repred_mode=hint_repred_mode)(features_list,
                                                                         repred,
                                                                         algorithm_index,
                                                                         return_hints,
                                                                         return_all_outputs)

        # print('dfa_baselines line 186~')
        self.net_fn = hk.transform(_use_net)

        # func, static_arg, extra_args = jax.jit, 'static_argnums', {}
        # extra_args[static_arg] = 3
        self.jitted_grad = jax.jit(self._compute_grad, static_argnums=3)
        # self.jitted_grad = jax.jit(self._compute_grad)
        # extra_args[static_arg] = 4
        self.jitted_feedback = jax.jit(self._feedback,
                                       donate_argnums=[0, 3],
                                       static_argnums=4)
        # extra_args[static_arg] = [3, 4, 5]
        self.jitted_predict = jax.jit(self._predict, static_argnums=[3, 4, 5])
        # extra_args[static_arg] = [3, 4]
        self.jitted_accum_opt_update = jax.jit(baselines.accum_opt_update,
                                               donate_argnums=[0, 2],
                                               static_argnums=[3, 4])

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
                                       return_hints=False,
                                       return_all_outputs=False)
        # print('dfa_baseline line 211')
        self.opt_state = self.opt.init(self.params)
        # We will use the optimizer state skeleton for traversal when we
        # want to avoid updating the state of params of untrained algorithms.
        self.opt_state_skeleton = self.opt.init(jnp.zeros(1))

    @property
    def params(self):
        if self._device_params is None:
            return None
        return jax.device_get(self._device_params)

    @params.setter
    def params(self, params):
        self._device_params = jax.device_put(params)

    @property
    def opt_state(self):
        # print('dfa_baselines line 236~ in property opt_state')
        if self._device_opt_state is None:
            return None
        return jax.device_get(self._device_opt_state)

    @opt_state.setter
    def opt_state(self, opt_state):
        self._device_opt_state = jax.device_put(opt_state)

    def _compute_grad(self, params, rng_key, feedback, algorithm_index):
        # print('dfa_baselines line 246~ in _compute_grad')
        lss, grads = jax.value_and_grad(self._loss)(
            params, rng_key, feedback, algorithm_index)
        return lss, grads

    def _feedback(self, params, rng_key, feedback, opt_state, algorithm_index):
        # print('dfa_baselines line 252~ in _feedback')
        lss, grads = jax.value_and_grad(self._loss)(
            params, rng_key, feedback, algorithm_index)
        params, opt_state = self._update_params(params, grads, opt_state,
                                                algorithm_index)
        return lss, params, opt_state

    def _predict(self, params, rng_key: hk.PRNGSequence, features: _Features,
                 algorithm_index: int, return_hints: bool,
                 return_all_outputs: bool):
        # print('dfa_baselines line 264~ in _predict')
        outs, hint_preds = self.net_fn.apply(
            params, rng_key, [features],
            repred=True, algorithm_index=algorithm_index,
            return_hints=return_hints,
            return_all_outputs=return_all_outputs)
        outs = decoders.postprocess(self._spec[algorithm_index],
                                    outs,
                                    sinkhorn_temperature=0.1,
                                    sinkhorn_steps=50,
                                    hard=True,
                                    )
        return outs, hint_preds

    def compute_loss(
            self,
            rng_key: hk.PRNGSequence,
            feedback: _Feedback,
            algorithm_index: Optional[int] = None,
    ) -> float:
        """Compute gradients."""

        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        assert algorithm_index >= 0
        # print('dfa_baselines line 290~ in compute_loss')
        # Calculate gradients.
        loss, _ = self.jitted_grad(
            self._device_params, rng_key, feedback, algorithm_index)
        return loss

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
        loss, grads = self.jitted_grad(
            self._device_params, rng_key, feedback, algorithm_index)
        return loss, grads

    def feedback(self, rng_key: hk.PRNGSequence,
                 feedback: _Feedback,
                 algorithm_index=None) -> float:
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        # Calculate and apply gradients.
        # print('dfa_baselines line 329~ in feedback')
        loss, self._device_params, self._device_opt_state = self.jitted_feedback(
            self._device_params, rng_key, feedback,
            self._device_opt_state, algorithm_index)
        return loss

    def predict(self, rng_key: hk.PRNGSequence, features: _Features,
                algorithm_index: Optional[int] = None,
                return_hints: bool = False,
                return_all_outputs: bool = False):
        """Model inference step."""
        if algorithm_index is None:
            assert len(self._spec) == 1
            algorithm_index = 0
        # print('dfa_baselines line 346~ in predict')
        return self.jitted_predict(
            self._device_params, rng_key, features,
            algorithm_index,
            return_hints,
            return_all_outputs)

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
                                                  lengths=hint_len)
        return total_loss

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

    def update_model_params_accum(self, grads) -> None:
        grads = jax.device_put(grads)
        # print(f'dfa_baseline line 383, grads on {grads.device()}')
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
