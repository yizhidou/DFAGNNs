from typing import Dict, List, Tuple
from clrs._src import probing
from clrs._src import specs
from clrs._src import losses

import chex
import jax
import jax.numpy as jnp

_DataPoint = probing.DataPoint
_Type = specs.Type
_OutputClass = specs.OutputClass

_chex_Array = chex.Array


def trace_o_loss(truth: _DataPoint,
                 pred: _chex_Array) -> float:
    """Output loss for full-sample training."""
    if truth.type_ == specs.Type.MASK:
        loss = (jnp.maximum(pred, 0) - pred * truth.data +
                jnp.log1p(jnp.exp(-jnp.abs(pred))))
        mask = (truth.data != _OutputClass.MASKED).astype(jnp.float32)
        total_loss = jnp.sum(loss * mask) / jnp.sum(mask)
    else:
        assert truth.type_ == specs.Type.CATEGORICAL
        # print(f'dfa_losses line 27, softmax(pred) = {jax.nn.log_softmax(pred)}')
        masked_truth = truth.data * (truth.data != _OutputClass.MASKED).astype(
            jnp.float32)
        total_loss = (-jnp.sum(masked_truth * jax.nn.log_softmax(pred)) /
                      jnp.sum(truth.data == _OutputClass.POSITIVE))

    return total_loss


def trace_h_loss(
        truth: _DataPoint,
        preds: _chex_Array,
        lengths: _chex_Array,
        take_hint_as_outpt: bool,
        verbose: bool = False,
):
    """Hint loss for full-sample training."""
    total_loss = 0.
    verbose_loss = {}
    # length = truth.data.shape[0] - 1
    length = truth.data.shape[0] - 1 if take_hint_as_outpt else truth.data.shape[0] - 2
    print(f'dfa_loss line 48, shape of preds = {preds.shape} \nshape of truth is: {truth.data[1:].shape}; shape of preds = {preds.shape}')
    loss, mask = _trace_h_loss(
        # truth_data=truth.data[1:],
        truth_data=truth.data[1:] if take_hint_as_outpt else truth.data[1:-1],
        truth_type=truth.type_,
        pred=preds,
    )
    mask *= losses._is_not_done_broadcast(lengths, jnp.arange(length)[:, None], loss)
    loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), losses.EPS)
    if verbose:
        verbose_loss['loss_' + truth.name] = loss
    else:
        total_loss += loss

    return verbose_loss if verbose else total_loss


def _trace_h_loss(
        truth_data: _chex_Array,
        truth_type: str,
        pred: _chex_Array) -> Tuple[_chex_Array, _chex_Array]:
    """Hint loss helper."""
    # print(f'dfa_losses line 58: truth_data: {truth_data.shape}; pred: {pred.shape}')
    if truth_type == specs.Type.MASK:
        loss = (jnp.maximum(pred, 0) - pred * truth_data +
                jnp.log1p(jnp.exp(-jnp.abs(pred))))
        mask = (truth_data != _OutputClass.MASKED).astype(
            jnp.float32)  # pytype: disable=attribute-error  # numpy-scalars
    else:
        assert truth_type == specs.Type.CATEGORICAL
        loss = -jnp.sum(truth_data * jax.nn.log_softmax(pred), axis=-1)
        mask = jnp.any(truth_data == _OutputClass.POSITIVE, axis=-1).astype(
            jnp.float32)
    return loss, mask
