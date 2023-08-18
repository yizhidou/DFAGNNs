from typing import Dict, Optional
from clrs._src import decoders
from clrs._src import specs
from clrs._src import yzd_probing

import chex
import haiku as hk
import jax
import jax.numpy as jnp

_DenseArray = chex.Array
_DataPoint = yzd_probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type


def sparse_postprocess(spec: _Spec, preds: Dict[str, _DenseArray],
                       sinkhorn_temperature: float,
                       sinkhorn_steps: int,
                       hard: bool) -> Dict[str, _DataPoint]:
    """Postprocesses decoder output.

    This is done on outputs in order to score performance, and on hints in
    order to score them but also in order to feed them back to the model.
    At scoring time, the postprocessing mode is "hard", logits will be
    arg-maxed and masks will be thresholded. However, for the case of the hints
    that are fed back in the model, the postprocessing can be hard or soft,
    depending on whether we want to let gradients flow through them or not.

    Args:
      spec: The spec of the algorithm whose outputs/hints we are postprocessing.
      preds: Output and/or hint predictions, as produced by decoders.
      sinkhorn_temperature: Parameter for the sinkhorn operator on permutation
        pointers.
      sinkhorn_steps: Parameter for the sinkhorn operator on permutation
        pointers.
      hard: whether to do hard postprocessing, which involves argmax for
        MASK_ONE, CATEGORICAL and POINTERS, thresholding for MASK, and stop
        gradient through for SCALAR. If False, soft postprocessing will be used,
        with softmax, sigmoid and gradients allowed.
    Returns:
      The postprocessed `preds`. In "soft" post-processing, POINTER types will
      change to SOFT_POINTER, so encoders know they do not need to be
      pre-processed before feeding them back in.
    """
    result = {}
    for name in preds.keys():
        _, loc, t = spec[name]
        new_t = t
        data = preds[name]
        if t == _Type.SCALAR:
            if hard:
                data = jax.lax.stop_gradient(data)
        elif t == _Type.MASK:
            if hard:
                data = (data > 0.0) * 1.0
            else:
                data = jax.nn.sigmoid(data)
        elif t in [_Type.MASK_ONE, _Type.CATEGORICAL]:
            cat_size = data.shape[-1]
            if hard:
                best = jnp.argmax(data, -1)
                data = hk.one_hot(best, cat_size)
            else:
                data = jax.nn.softmax(data, axis=-1)
        elif t == _Type.POINTER:
            if hard:
                data = jnp.argmax(data, -1).astype(float)
            else:
                data = jax.nn.softmax(data, -1)
                new_t = _Type.SOFT_POINTER
        elif t == _Type.PERMUTATION_POINTER:
            # Convert the matrix of logits to a doubly stochastic matrix.
            data = log_sinkhorn(
                x=data,
                steps=sinkhorn_steps,
                temperature=sinkhorn_temperature,
                zero_diagonal=True,
                noise_rng_key=None)
            data = jnp.exp(data)
            if hard:
                data = jax.nn.one_hot(jnp.argmax(data, axis=-1), data.shape[-1])
        else:
            raise ValueError("Invalid type")
        result[name] = probing.DataPoint(
            name=name, location=loc, type_=new_t, data=data)

    return result
