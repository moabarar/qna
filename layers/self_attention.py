from typing import Callable, Optional, Any, Tuple, Union

import jax
import jax.random as random
from flax import linen as nn
from jax import numpy as jnp

from .positional_encoding import RelativePositionalEmbedding

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

DEFAULT_KERNEL_INIT = nn.initializers.lecun_normal()
DEFAULT_BIAS_INIT = nn.initializers.zeros
MINUS_INF = -1e6


class MultiHeadSelfAttention(nn.Module):
  features: int
  heads: int
  output_features: Optional[int] = None
  train: Optional[bool] = None
  use_bias: bool = False
  kernel_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  dtype: Dtype = jnp.float32
  attention_dropout: float = 0.0

  # Attributes related to Relative Positional Embedding
  use_relative_pe: bool = True
  rpe_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  window_size: Optional[Union[Tuple[int], int]] = None

  @nn.compact
  def __call__(self, inputs, mask: Optional[bool] = None, train: Optional[bool] = None):
    train = nn.module.merge_param('train', self.train, train)
    qkv = nn.Dense(3 * self.features,
                   dtype=self.dtype,
                   kernel_init=self.kernel_init,
                   name='to_qkv',
                   use_bias=self.use_bias)(inputs)
    qkv = jnp.reshape(qkv, [*qkv.shape[0:-1], self.heads, 3 * self.features // self.heads])
    q, k, v = jnp.split(qkv, 3, axis=-1)

    bias = None
    if self.use_relative_pe:
      assert self.window_size is not None, 'Need to provide window size for relative positional embedding'
      bias = RelativePositionalEmbedding(heads=self.heads,
                                         window_size=self.window_size,
                                         dtype=self.dtype,
                                         rpe_init=self.rpe_init)()
    if mask is not None:
      mask = jnp.asarray(mask, self.dtype) * MINUS_INF
      bias = mask if bias is None else bias + mask

    attn_weights = nn.attention.dot_product_attention_weights(q,
                                                              k,
                                                              dropout_rate=self.attention_dropout,
                                                              deterministic=not train,
                                                              dtype=self.dtype,
                                                              bias=bias)

    # return weighted sum over values for each query position
    out_per_head = jnp.einsum('...hqk,...khd->...qhd', attn_weights, v)
    out_concat = jnp.reshape(out_per_head, [*out_per_head.shape[:-2], self.features])
    out = nn.Dense(self.output_features if self.output_features else self.features,
                   dtype=self.dtype,
                   kernel_init=self.kernel_init,
                   bias_init=self.bias_init,
                   name='to_out',
                   use_bias=self.use_bias)(out_concat)

    return out
