"""Implementation of different Positional Encodings."""
from typing import Any, Callable, Tuple, Optional, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import random

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

DEFAULT_KERNEL_INIT = nn.initializers.lecun_normal()
DEFAULT_BIAS_INIT = nn.initializers.zeros


class RelativePositionalEmbedding(nn.Module):
  # Based on https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
  heads: int
  window_size: Union[int, Tuple[int]]
  dtype: Dtype = jnp.float32
  rpe_init: Optional[Callable[[random.PRNGKey, Shape, Dtype], Array]] = None
  scale_rpe: Optional[float] = None

  @nn.compact
  def __call__(self):
    window_size = (self.window_size, self.window_size) if isinstance(self.window_size, int) else self.window_size
    _normal_init_fn = lambda key, shape: random.truncated_normal(key, -2., 2.0, shape, jnp.float32) * 0.02
    pos_embedding_init = self.rpe_init if self.rpe_init is not None else _normal_init_fn
    relative_position_bias_table = self.param('rpe_bias', pos_embedding_init,
                                              [(2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                                               self.heads])
    relative_position_bias_table = jnp.asarray(relative_position_bias_table, self.dtype)
    # get pair-wise relative position index for each token inside the
    coords_h = jnp.arange(window_size[0])
    coords_w = jnp.arange(window_size[1])
    coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing='ij'))  # 2, Wh, Ww
    coords_flatten = coords.reshape([coords.shape[0], -1])  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
    R = relative_coords[:, :, 0]
    R = R + window_size[0] - 1  # shift to start from 0
    R = (2 * window_size[1] - 1) * R
    C = relative_coords[:, :, 1]
    C = C + window_size[1] - 1
    relative_position_index = R + C  # Wh*Ww, Wh*Ww
    rpe_bias = relative_position_bias_table[relative_position_index.reshape(-1)]
    rpe_bias = rpe_bias.reshape([window_size[0] * window_size[1],
                                 window_size[0] * window_size[1], -1])
    rpe_bias = rpe_bias.transpose([2, 0, 1])
    if self.scale_rpe is not None:
      rpe_bias = rpe_bias / self.scale_rpe
    return rpe_bias
