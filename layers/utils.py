"""Implementation of different stochastic Modules (e.g., DropPath)."""

from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
import jax.random
from jax import random

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

DEFAULT_KERNEL_INIT = nn.initializers.lecun_normal()
DEFAULT_BIAS_INIT = nn.initializers.zeros


class DropPath(nn.Module):
  """Create a stochastic depth layer.

  Follows dropout implementation from
  flax/linen/stochastic.py

    Attributes:
      rate: the drop probability.  (_not_ the keep rate!)
      deterministic: if false the inputs are scaled by `1 / (1 - rate)` and
        masked, whereas if true, no mask is applied and the inputs are returned
        as is.
  Copied from: https://github.com/google-research/nested-transformer/blob/main/libml/attn_utils.py
  """
  rate: float
  deterministic: Optional[bool] = None

  @nn.compact
  def __call__(self, inputs):
    if self.rate == 0.:
      return inputs
    keep_prob = 1. - self.rate
    if self.deterministic:
      return inputs
    else:
      # just use the same set of naming with dropout
      rng = self.make_rng("dropout")
      mask_shape = [inputs.shape[0]] + [1 for _ in inputs.shape[1:]]
      mask = jax.random.bernoulli(rng, p=keep_prob, shape=mask_shape)
      mask = jnp.tile(mask, [1] + list(inputs.shape[1:]))
      scaled_inputs = inputs / keep_prob
      scaled_inputs = scaled_inputs.astype(inputs.dtype)
      return jax.lax.select(mask, scaled_inputs, jnp.zeros_like(inputs))


class LayerScale(nn.Module):
  """LayerScale was defined in CaiT: https://arxiv.org/abs/2103.17239"""
  features: int
  scale_init: float
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, inputs):
    _init_fn = lambda key, shape: jnp.ones(shape, dtype=jnp.float32) * self.scale_init
    scale_param = self.param('scale_param', _init_fn, (self.features,))
    scale_param = jnp.asarray(scale_param, self.dtype)
    return inputs * scale_param


class MlpBlock(nn.Module):
  """MLP blocks.

  Based on Flax implementation:
  https://github.com/google/flax/blob/main/examples/lm1b/models.py
  """
  mlp_dim: Optional[int] = None
  out_dim: Optional[int] = None
  dense_fn: Callable = nn.Dense  # pylint: disable=g-bare-generic
  activation_fn: Any = nn.gelu
  proj_drop: float = 0.0
  kernel_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  use_bias: bool = True
  train: bool = True
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, x):
    actual_out_dim = (x.shape[-1] if self.out_dim is None else self.out_dim)
    mlp_dim = x.shape[-1] if self.mlp_dim is None else self.mlp_dim
    x = self.dense_fn(mlp_dim,
                      dtype=self.dtype,
                      kernel_init=self.kernel_init,
                      use_bias=self.use_bias,
                      bias_init=self.bias_init)(x)
    x = self.activation_fn(x)
    x = nn.Dropout(self.proj_drop, deterministic=not self.train)(x)
    x = self.dense_fn(actual_out_dim,
                      dtype=self.dtype,
                      kernel_init=self.kernel_init,
                      use_bias=self.use_bias,
                      bias_init=self.bias_init)(x)
    x = nn.Dropout(self.proj_drop, deterministic=not self.train)(x)
    return x
