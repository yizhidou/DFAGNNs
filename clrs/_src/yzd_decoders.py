from typing import Dict, Optional
from clrs._src import decoders
from clrs._src import specs
from clrs._src import dfa_utils

import chex
import haiku as hk
import jax
import jax.numpy as jnp

_chex_Array = chex.Array
# _DataPoint = yzd_probing.DataPoint
_Location = specs.Location
_Spec = specs.Spec
_Stage = specs.Stage
_Type = specs.Type


def decode_fts(decoders,
               h_t: _chex_Array):
    decoded_trace_o = jnp.squeeze(decoders[0](h_t), -1)
    decoded_trace_h = None
    if 'trace_h' in decoders:
        decoded_trace_h = jnp.squeeze(decoders[0](h_t), -1)
    return decoded_trace_h, decoded_trace_o

