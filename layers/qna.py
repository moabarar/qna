from typing import Callable, Sequence, Union, Optional, Any, Tuple

import jax
import jax.ops
import jax.random as random
from flax import linen as nn
from jax import lax
from jax import numpy as jnp

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

DEFAULT_KERNEL_INIT = nn.initializers.lecun_normal()
DEFAULT_BIAS_INIT = nn.initializers.zeros


def _normalize_and_reshape_query(q, dtype, heads, unit_norm, depth_scale, max_norm=None, normalize_stop_grads=False):
  """Normalizes the query and prepares it for attention computation."""
  assert q.ndim > 0
  assert max_norm is None or (max_norm is not None and not unit_norm)
  q = jnp.asarray(q, dtype=dtype)
  newshape = [heads, q.shape[-1] // heads]
  if q.ndim > 1:
    newshape = [*q.shape[:-1], *newshape]
  q = jnp.reshape(q, newshape)
  if unit_norm:
    q_norm = jnp.linalg.norm(q, ord=2, axis=-1, keepdims=True)
    if normalize_stop_grads:
      q_norm = jax.lax.stop_gradient(q_norm)
    q = q / (q_norm + 1e-6)
  if max_norm is not None:
    assert not unit_norm, 'Limiting queries norm cannot be enforced while unit_norm is set.'
    q_norm = jnp.linalg.norm(q, ord=2, axis=1, keepdims=True)
    q_norm_scale = max_norm / (q_norm + 1e-6)
    q_norm_scale = jnp.clip(q_norm_scale, a_max=1.0)
    q = q * q_norm_scale
  if depth_scale:
    depth = q.shape[-1]
    q = q / jnp.sqrt(depth).astype(dtype)
  return q


class FusedKQnA(nn.Module):
  features: int
  heads: int
  kernel_size: Union[Sequence[int], int]
  stride: Union[Sequence[int], int]
  padding: Union[Sequence[int], int]
  out_features: Optional[int] = None
  use_bias: bool = False
  pos_embedding_type: str = 'relative_pos'
  kernel_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  pos_embedding_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  query_init: Optional[Callable[[random.PRNGKey, Shape, Dtype], Array]] = None
  dtype: Dtype = jnp.float32
  attn_scale: str = 'normal'
  normalize_q: bool = True  # For better stability
  qna_normalize_stop_grads: bool = False
  n_queries: int = 2
  precision: Optional[lax.Precision] = None
  scale_rpe_bias: bool = False

  def _get_query(self):
    """Defines the queries parameters and reshape then into [..., N_Queries, Heads, Head_dim].
    Queries are also scaled by SQRT(Head_dim) for stability (proposed in https://arxiv.org/abs/1706.03762).

    Returns: Queries after preprocessing with shape [..., N_Queries, Heads, Head_dim].

    """

    def querey_init_normalized_(key, shape, dtype=jnp.float32):
      stddev = jnp.sqrt(1.0 / (self.features // self.heads))
      return random.truncated_normal(key, -2, 2, shape, dtype) * stddev

    q_init = querey_init_normalized_ if self.query_init is None else self.query_init
    q = self.param('query', q_init, (self.n_queries, self.features))
    q = jnp.asarray(q, self.dtype)
    q = _normalize_and_reshape_query(q, self.dtype, self.heads, self.normalize_q, depth_scale=True,
                                     normalize_stop_grads=self.qna_normalize_stop_grads)
    return q

  def _compute_QK_scores(self, q, x):
    """Computes the QK dot product in fused manner.
    Since the queries are shared across windows, we compute (Q*W_k^T)X^T for better memory utilization.

    :param q: The leared queries of shape [..., N_Queries, Heads, Head_dim]
    :param x: The input features [B, H, W, C]
    :return: The query-key dot product for each query head [B, H, W, N_Queries, Heads]
    """
    # q = [Nq, h, d]
    Wk = self.param('Wk', self.kernel_init, [x.shape[-1], self.features])
    Wk = jnp.asarray(Wk, self.dtype)
    # WK = [D_in, h, D]
    Wk = Wk.reshape([-1, self.heads, self.features // self.heads])
    qWk = jnp.einsum('Bqhd,Dhd->BDqh', q, Wk, precision=None)
    qWkx = jnp.einsum('BHWD,BDqh->BHWqh', x, qWk, precision=None)
    if self.use_bias:
      Wk_b = self.param('Wk_b', self.bias_init, [self.heads, self.features // self.heads])
      Wk_b = jnp.asarray(Wk_b, self.dtype)
      qWk_b = jnp.einsum('hd,Bqhd->Bqh', Wk_b, q, precision=None)[:, None, None, ...]
      qWkx = qWkx + qWk_b

    return qWkx

  def _get_query_attention_scale(self):
    """The query attention scale defines how to aggregate different queries attention into one attention.

    :return: The weights that will be used to apply the scaling
    """
    if self.attn_scale == 'normal':
      _normal_init_fn = lambda key, shape: random.normal(key, shape, jnp.float32) * 0.02
      attn_scale = self.param('attn_scale_weights', _normal_init_fn,
                              (1, 1, self.kernel_size ** 2, self.n_queries * self.heads))
      attn_scale = jnp.asarray(attn_scale, self.dtype)
      attn_scale = attn_scale.reshape(
        (self.kernel_size, self.kernel_size, 1, self.n_queries * self.heads))  # Backward compatibility
    elif self.attn_scale == 'normal_elementwise':
      _normal_init_fn = lambda key, shape: random.normal(key, shape, jnp.float32) * 0.02
      attn_scale = self.param('attn_scale_weights', _normal_init_fn,
                              (1, 1, 1, self.n_queries * self.heads))
      attn_scale = jnp.asarray(attn_scale, self.dtype)
      attn_scale = attn_scale.reshape((1, 1, 1, self.n_queries * self.heads))  # Backward compatibility
    elif self.attn_scale == 'posneg':
      attn_scale = jnp.concatenate([jnp.ones((1, 1, 1, self.n_queries * self.heads // 2)),
                                    jnp.ones((1, 1, 1, self.n_queries * self.heads // 2)) * -1], axis=3)
      attn_scale = jnp.asarray(attn_scale, self.dtype)
      attn_scale = attn_scale.reshape((1, 1, 1, self.n_queries * self.heads))  # Backward compatibility
    elif self.attn_scale == 'ones':
      _pos_init_fn = lambda key, shape: jnp.ones(shape, jnp.float32)
      attn_scale = self.param('attn_scale_pos', _pos_init_fn, (1, 1, self.kernel_size ** 2, self.n_queries, self.heads))
      attn_scale = jnp.asarray(attn_scale, self.dtype)
      attn_scale = attn_scale.reshape(
        (self.kernel_size, self.kernel_size, 1, self.n_queries * self.heads))  # Backward compatibility
    elif self.attn_scale == 'avg':
      attn_scale = jnp.ones((1, self.n_queries, self.heads), jnp.float32) / self.n_queries
      attn_scale = jnp.asarray(attn_scale, self.dtype)
      attn_scale = attn_scale.reshape(
        (1, 1, 1, self.n_queries * self.heads))  # Backward compatibility
    elif self.attn_scale == 'none':
      attn_scale = None
    else:
      raise ValueError(f'Scale {self.attn_scale} not supported!')

    return attn_scale

  def _get_relative_bias(self, q):
    """Return the relative positional embedding that will be added to the QK scores before applying the softmax"""
    if self.pos_embedding_type == 'relative_pos':
      _normal_init_fn = lambda key, shape: random.truncated_normal(key, -2., 2.0, shape, jnp.float32) * 0.02
      pos_embedding_init = _normal_init_fn if self.pos_embedding_init is None else self.pos_embedding_init
      relative_pos_embedding = self.param('rpe_bias', pos_embedding_init,
                                          (self.kernel_size ** 2, self.n_queries * self.heads))
      if self.scale_rpe_bias:
        relative_pos_embedding = relative_pos_embedding / jnp.sqrt(self.features // self.heads)
      relative_pos_embedding = jnp.asarray(relative_pos_embedding, self.dtype)
      relative_pos_embedding = relative_pos_embedding.reshape(
        (self.kernel_size, self.kernel_size, 1, self.n_queries * self.heads))  # Backward compatibility
    elif self.pos_embedding_type == 'contextual_pos':
      # Defined in: https://arxiv.org/abs/2107.14222
      # q is already scaled by SQRT(Head_dim)...
      relative_pos_embedding = nn.DenseGeneral(features=[self.kernel_size ** 2, self.heads], axis=[-2, -1],
                                               name='rpe_bias', dtype=self.dtype)(q)
      relative_pos_embedding = relative_pos_embedding.reshape(
        (self.kernel_size, self.kernel_size, 1, self.n_queries * self.heads))  # Backward compatibility
    elif self.pos_embedding_type == 'absolute_pos':
      pos_embedding = self.param('absolute_pos_embedding', self.pos_embedding_init,
                                 (self.kernel_size ** 2, self.n_queries, self.features))
      pos_embedding = jnp.asarray(pos_embedding, self.dtype)
      pos_embedding = jnp.reshape(pos_embedding,
                                  [self.kernel_size ** 2, self.n_queries, self.heads, self.features // self.heads])
      relative_pos_embedding = jnp.einsum('Bqhd,...kqhd->...kqh', q, pos_embedding)
      relative_pos_embedding = jnp.reshape(relative_pos_embedding,
                                           (self.kernel_size, self.kernel_size, 1,
                                            self.n_queries * self.heads))  # Backward compatibility
    elif self.pos_embedding_type == 'none':
      relative_pos_embedding = None
    else:
      assert False, f'The following positional {self.pos_embedding_type} is no supported'

    return relative_pos_embedding

  def _compute_attention(self, cost, v, rpe=None, attn_scale=None):
    """Compute the attention in memory efficent manner (see paper: https://arxiv.org/abs/2112.11435)"""
    B, H, W, _ = v.shape
    cost_exp = jnp.exp(cost)
    v_cost_exp = cost_exp[..., jnp.newaxis] * jnp.reshape(v, [B, H, W, 1, self.heads,
                                                              self.features // self.heads])  # [B, H, W, self.n_queries, h, d]
    v_cost_exp = jnp.reshape(v_cost_exp, [B, H, W, self.n_queries * self.heads * self.features // self.heads])
    rpe_exp = jnp.exp(rpe)
    if attn_scale is not None:
      summation_kernel = rpe_exp * attn_scale  # [self.kernel_size, self.kernel_size, 1, self.n_queries*h]
    else:
      summation_kernel = rpe_exp
    summation_kernel = jnp.repeat(summation_kernel, repeats=self.features // self.heads, axis=-1)
    I = v_cost_exp
    sum_v_cost_exp = jax.lax.conv_general_dilated(I,
                                                  summation_kernel,
                                                  window_strides=[self.stride, self.stride],
                                                  padding=[(self.padding, self.padding), (self.padding, self.padding)],
                                                  feature_group_count=self.n_queries * self.features,
                                                  dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                                  precision=self.precision,
                                                  )
    h_out, w_out = sum_v_cost_exp.shape[1], sum_v_cost_exp.shape[2]
    sum_v_cost_exp = jnp.reshape(sum_v_cost_exp,
                                 [B, h_out, w_out, self.n_queries, self.heads, self.features // self.heads])
    I = cost_exp.reshape([B, H, W, -1])
    summation_kernel = rpe_exp
    sum_cost_exp = jax.lax.conv_general_dilated(I,
                                                summation_kernel,
                                                window_strides=[self.stride, self.stride],
                                                padding=[(self.padding, self.padding), (self.padding, self.padding)],
                                                feature_group_count=self.n_queries * self.heads,
                                                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
                                                ).reshape([B, h_out, w_out, self.n_queries, self.heads, 1])
    out = sum_v_cost_exp / sum_cost_exp
    out = jnp.sum(out, axis=-3).reshape([B, h_out, w_out, self.features])
    return out

  @nn.compact
  def __call__(self, x):
    q = self._get_query()
    q = jnp.broadcast_to(q, [x.shape[0], *q.shape])  # q = [B, Nq, h, d]

    # Compute the QK Scores that will be used to define the attention weights.
    QK_score = self._compute_QK_scores(q, x)
    # Compute V
    V = nn.Dense(self.features,
                 use_bias=self.use_bias,
                 kernel_init=self.kernel_init,
                 bias_init=self.bias_init,
                 dtype=self.dtype,
                 name='to_v')(x)

    attention_scale = self._get_query_attention_scale()
    relative_bias = self._get_relative_bias(q)

    out_per_head_concat = self._compute_attention(QK_score, V, rpe=relative_bias, attn_scale=attention_scale)
    out = nn.Dense(self.features if self.out_features is None else self.out_features,
                   use_bias=self.use_bias,
                   kernel_init=self.kernel_init,
                   bias_init=self.bias_init,
                   dtype=self.dtype,
                   name='to_out')(out_per_head_concat)
    return out
