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

    loss = (jnp.maximum(pred, 0) - pred * truth.data +
            jnp.log1p(jnp.exp(-jnp.abs(pred))))
    mask = (truth.data != _OutputClass.MASKED).astype(jnp.float32)
    total_loss = jnp.sum(loss * mask) / jnp.sum(mask)

    return total_loss


def trace_h_loss(
        truth: _DataPoint,
        preds: List[_chex_Array],
        lengths: _chex_Array,
        verbose: bool = False,
):
    """Hint loss for full-sample training."""
    total_loss = 0.
    verbose_loss = {}
    length = truth.data.shape[0] - 1

    loss, mask = _trace_h_loss(
        truth_data=truth.data[1:],
        pred=jnp.stack(preds),
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
        pred: _chex_Array) -> Tuple[_chex_Array, _chex_Array]:
    """Hint loss helper."""
    loss = (jnp.maximum(pred, 0) - pred * truth_data +
            jnp.log1p(jnp.exp(-jnp.abs(pred))))
    mask = (truth_data != _OutputClass.MASKED).astype(
        jnp.float32)  # pytype: disable=attribute-error  # numpy-scalars
    return loss, mask
