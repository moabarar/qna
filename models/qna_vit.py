"""The implementation of QnA-ViT (https://arxiv.org/pdf/2112.11435.pdf)"""
import functools
from typing import Any, Callable, Tuple, Optional, List, Union

import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import numpy as np
from jax import random

import layers

# Store models in dictionary allow retrieval by name.
MODELS = {}


def register(f):
  MODELS[f.__name__] = f
  return f


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

DEFAULT_KERNEL_INIT = nn.initializers.lecun_normal()
DEFAULT_BIAS_INIT = nn.initializers.zeros

RPE_STR_REPRESENTATION = 'relative_pos'
ABS_POS_STR_REPRESENTATION = 'absolute_pos'
CONTEXT_POS_STR_REPRESENTATION = 'contextual_pos'


class StemBlock(nn.Module):
  """A Stem block for initial patch tokenization.
  Two types of STEM blocks are supported, either 'patch' style (similar to ViT) or 'resnet'"""
  features: int
  stem_type: str = 'patch'
  activation_fn: Optional[Callable[[Array], Array]] = nn.gelu
  patch_size: int = 4
  use_bias: bool = False
  # === Common Parameters ===
  dtype: int = jnp.float32
  train: Optional[bool] = None
  kernel_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT

  @nn.compact
  def __call__(self, x, train: Optional[bool] = None):
    train = nn.module.merge_param('train', self.train, train)
    patch_size = self.patch_size
    if self.stem_type == 'resnet':
      x = nn.Conv(features=self.features, kernel_size=(7, 7), strides=(2, 2),
                  padding='SAME',
                  name='conv_7x7',
                  use_bias=False,
                  dtype=self.dtype,
                  kernel_init=self.kernel_init,
                  bias_init=self.bias_init)(x)
      x = nn.BatchNorm(dtype=self.dtype, use_running_average=not train, momentum=0.9)(x)
      x = self.activation_fn(x)
      x = nn.max_pool(x, (3, 3), strides=(2, 2), padding="SAME")
    elif self.stem_type == 'patch':
      x = nn.Conv(features=self.features, kernel_size=(patch_size, patch_size),
                  strides=(patch_size, patch_size),
                  padding='SAME',
                  name='conv_1x1',
                  use_bias=self.use_bias,
                  dtype=self.dtype,
                  kernel_init=self.kernel_init,
                  bias_init=self.bias_init)(x)
    else:
      raise ValueError(f'Available stem type are [\'patch\', \'resnet\']. Got: {self.stem_type}.')
    return x


class BaseBlock(nn.Module):
  spatial_mixing_fn_maker: Callable[[], Callable[[Array], Array]]
  spatial_mixing_skip_fn: Any = None
  drop_path: float = 0.0
  layer_scale: Optional[float] = None
  # === Common Parameters ===
  dtype: int = jnp.float32
  train: Optional[bool] = None
  kernel_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  use_bias: bool = True
  layer_scale: Optional[float] = None

  @nn.compact
  def __call__(self, x):
    norm_fn = functools.partial(nn.LayerNorm, dtype=self.dtype)
    skip = x
    if self.spatial_mixing_skip_fn is not None:
      # To support skip connection with strided QnA we apply 1x1 conv with stride 2 on the skip path.
      skip = self.spatial_mixing_skip_fn()(skip)

    # Spatial Mixing:
    x = self.spatial_mixing_fn_maker()(norm_fn()(x))
    if self.layer_scale is not None:
      x = layers.LayerScale(scale_init=self.layer_scale,
                            features=x.shape[-1],
                            dtype=self.dtype)(x)

    if self.spatial_mixing_skip_fn is None:
      x = layers.DropPath(rate=self.drop_path, deterministic=not self.train)(x)

    x = skip = x + skip

    # Channel Mixing:
    x = layers.MlpBlock(mlp_dim=x.shape[-1] * 4,
                        out_dim=x.shape[-1],
                        proj_drop=0.0,
                        kernel_init=self.kernel_init,
                        bias_init=self.bias_init,
                        use_bias=self.use_bias,
                        dtype=self.dtype,
                        train=self.train, )(norm_fn()(x))
    if self.layer_scale is not None:
      x = layers.LayerScale(scale_init=self.layer_scale,
                            features=x.shape[-1],
                            dtype=self.dtype)(x)
    x = layers.DropPath(rate=self.drop_path, deterministic=not self.train)(x)
    x = x + skip

    return x


class AvgPool1x1Conv(nn.Module):
  features: int
  kernel_size: int
  stride: int
  kernel_init: Callable[
    [random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT
  use_bias: bool = False
  dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, x):
    x = nn.avg_pool(x, (self.kernel_size, self.kernel_size), strides=(self.stride, self.stride), padding='SAME')
    x = nn.Conv(features=self.features,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=[(0, 0) for _ in range(2)],
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
                use_bias=self.use_bias,
                dtype=self.dtype)(x)
    return x


class QnAStage(nn.Module):
  stage: int
  features: int
  downsample: bool
  config: ml_collections.ConfigDict
  drop_path: Union[List[float], float] = 0.0
  # Common Params:
  train: Optional[bool] = None
  dtype: int = jnp.float32
  kernel_init: Callable[
    [random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT

  def _get_qna_module(self, downsample=False):
    params = dict(features=self.features * (1 + int(downsample)),
                  heads=self.config.qna_heads[self.stage] * (1 + int(downsample)),
                  kernel_size=self.config.qna_receptive_field[self.stage],
                  stride=2 if downsample else 1,
                  padding=(self.config.qna_receptive_field[self.stage] - 1) // 2,
                  pos_embedding_type=self.config.qna_pos_embedding_type,
                  kernel_init=self.kernel_init,
                  pos_embedding_init=self.bias_init,
                  bias_init=self.bias_init,
                  dtype=self.dtype,
                  attn_scale=self.config.qna_attention_scale,
                  n_queries=self.config.qna_num_queries,
                  use_bias=False
                  )
    if self.config.qna_implementation == 'fused_efficient':
      params['normalize_q'] = self.config.qna_normalize_q[self.stage]
      params['qna_normalize_stop_grads'] = self.config.qna_normalize_stop_grads
      return functools.partial(layers.FusedKQnA, **params)
    else:
      raise ValueError(f'Not supported QnA implementation type {self.config.qna_implementation}')

  @nn.compact
  def __call__(self, x):
    n_qna = self.config.qna_layers[self.stage]
    drop_path = [self.drop_path for _ in range(n_qna)] if isinstance(self.drop_path, float) else self.drop_path

    for layer in range(n_qna):
      qna_layer_fn = self._get_qna_module(downsample=False)
      x = BaseBlock(spatial_mixing_fn_maker=qna_layer_fn,
                    drop_path=drop_path[layer],
                    dtype=self.dtype,
                    train=self.train,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    layer_scale=self.config.layer_scale,
                    name=f'QnA_Block_{layer + 1}')(x)

    if self.downsample:
      qna_layer_fn = self._get_qna_module(downsample=True)
      if self.config.get('qna_down_skip', 'conv1x1') == 'conv1x1':
        qna_skip_fn = functools.partial(nn.Conv, features=2 * self.features,
                                        kernel_size=(1, 1),
                                        strides=(2, 2),
                                        padding=[(0, 0) for _ in range(2)],
                                        kernel_init=self.kernel_init,
                                        use_bias=False,
                                        dtype=self.dtype)
      else:
        assert self.config.qna_down_skip == 'avgpool', f'Downsample skip connection {self.config.qna_down_skip} not supported'
        qna_skip_fn = functools.partial(AvgPool1x1Conv, features=2 * self.features,
                                        kernel_size=self.config.qna_receptive_field[self.stage],
                                        stride=2,
                                        kernel_init=self.kernel_init,
                                        bias_init=self.bias_init,
                                        use_bias=False,
                                        dtype=self.dtype)
      x = BaseBlock(spatial_mixing_fn_maker=qna_layer_fn,
                    spatial_mixing_skip_fn=qna_skip_fn,
                    drop_path=drop_path[layer + 1] if self.config.drop_path_downsample else 0.0,
                    dtype=self.dtype,
                    train=self.train,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    layer_scale=self.config.layer_scale,
                    name=f'QnA_Block_{n_qna + 1}_stride_2')(x)
    return x


class LSAStage(nn.Module):
  stage: int
  features: int
  config: ml_collections.ConfigDict
  drop_path: Union[List[float], float] = 0.0
  # Common Params:
  train: Optional[bool] = None
  dtype: int = jnp.float32
  kernel_init: Callable[
    [random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT

  def _split_to_windows(self, x, input_shape=None):
    assert x.ndim == 4, f'Input dimension should be 4, got ndim = {x.ndim} and shape {x.shape}'
    # TODO(marar) handle with padding if neceesary ?
    B, H, W, C = x.shape if input_shape is None else input_shape
    window_size = self.config.lsa_window[self.stage]
    assert H % window_size == 0 and W % window_size == 0, 'Input shape must be divisible by window size'
    x = jnp.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = jnp.transpose(x, [0, 1, 3, 2, 4, 5])
    x = jnp.reshape(x, [-1, window_size ** 2, C])
    return x

  def _merge_windows(self, x, input_shape):
    assert x.ndim == 3, f'Merging non-sequential input is not supported, ' \
                        f'ndim should be 3 got {x.ndim} with shape {x.shape}'
    window_size = self.config.lsa_window[self.stage]
    B, H, W, C = input_shape
    x = jnp.reshape(x, [B, H // window_size, W // window_size, window_size, window_size, C])
    x = jnp.transpose(x, [0, 1, 3, 2, 4, 5])
    x = jnp.reshape(x, [B, H, W, C])
    return x

  @nn.compact
  def __call__(self, x):
    n_lsa = self.config.lsa_layers[self.stage]
    if n_lsa == 0:  # Some stages may have no LSA layers (e.g., early stages)
      return x

    # Window partition batched images [B, H, W, C]
    # Save original shape before partition so we can later return input to original shape.
    original_shape = x.shape
    x = self._split_to_windows(x)

    drop_path = [self.drop_path for _ in range(n_lsa)] if isinstance(self.drop_path, float) else self.drop_path

    for layer in range(n_lsa):
      lsa_fn = functools.partial(layers.MultiHeadSelfAttention,
                                 features=self.features,
                                 heads=self.config.lsa_heads[self.stage],
                                 train=self.train,
                                 use_bias=False,
                                 kernel_init=self.kernel_init,
                                 bias_init=self.bias_init,
                                 dtype=self.dtype,
                                 attention_dropout=self.config.lsa_attention_dropout,
                                 use_relative_pe=self.config.lsa_pos_embedding_type == RPE_STR_REPRESENTATION,
                                 window_size=self.config.lsa_window[self.stage]
                                 )
      x = BaseBlock(spatial_mixing_fn_maker=lsa_fn,
                    drop_path=drop_path[layer],
                    dtype=self.dtype,
                    train=self.train,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                    layer_scale=self.config.layer_scale,
                    name=f'MSA_Block_{layer + 1}')(x)

    # Reshape sequence into original shape:
    x = self._merge_windows(x, original_shape)
    return x


class ClassificationHead(nn.Module):
  num_classes: int
  head_type: str
  num_heads: Optional[int] = None
  # Common Params:
  train: Optional[bool] = None
  dtype: int = jnp.float32
  kernel_init: Callable[
    [random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT

  @nn.compact
  def __call__(self, x):
    B, H, W, C = x.shape
    if self.head_type == 'gap':
      x = nn.LayerNorm(dtype=self.dtype)(x)
      x = jnp.reshape(x, [B, H * W, C])
      x = jnp.mean(x, axis=tuple(range(1, x.ndim - 1)))
    elif self.head_type == 'qnav1':
      assert self.num_heads, f'Need to define number of heads when using QnA classification head. Got: {self.num_heads}'
      assert H == W  # TODO(marar) Currently working with squared image - need to fix QnA to support non-squared kernels
      x = nn.LayerNorm(dtype=self.dtype)(x)
      x = layers.FusedKQnA(features=C,
                           stride=1,
                           heads=self.num_heads,
                           kernel_size=H,
                           dtype=self.dtype,
                           normalize_q=True,
                           n_queries=2,
                           padding=0,
                           kernel_init=self.kernel_init,
                           use_bias=False)(x)
      x = jnp.reshape(x, [B, C])
      x = x
    elif self.head_type == 'qnav2':
      assert self.num_heads, f'Need to define number of heads when using QnA classification head. Got: {self.num_heads}'
      assert H == W  # TODO(marar) Currently working with squared image - need to fix QnA to support non-squared kernels
      x = nn.LayerNorm(dtype=self.dtype)(x)
      skip = jnp.mean(jnp.reshape(x, [B, H * W, C]), axis=1)
      x = layers.FusedKQnA(features=C,
                           stride=1,
                           heads=self.num_heads,
                           kernel_size=H,
                           dtype=self.dtype,
                           normalize_q=True,
                           n_queries=2,
                           padding=0,
                           kernel_init=self.kernel_init,
                           use_bias=False)(x)
      x = jnp.reshape(x, [B, C])
      x = x + skip
    else:
      # TODO(moaba) add available heads to the documentation
      raise ValueError(f'Classifier head {self.head_type} no supported.')

    x = nn.Dense(features=self.num_classes,
                 name='head_logits',
                 kernel_init=self.kernel_init,
                 dtype=self.dtype)(x)

    return x


class QnAViT(nn.Module):
  num_classes: int
  config: ml_collections.ConfigDict
  train: Optional[bool] = None
  dtype: int = jnp.float32
  kernel_init: Callable[
    [random.PRNGKey, Shape, Dtype], Array] = DEFAULT_KERNEL_INIT
  bias_init: Callable[[random.PRNGKey, Shape, Dtype], Array] = DEFAULT_BIAS_INIT

  @nn.compact
  def __call__(self, x, train: Optional[bool] = None):
    train = nn.module.merge_param('train', self.train, train)
    # Common parameters that are used by QnA-Blocks and
    common_params = dict(dtype=self.dtype,
                         train=train,
                         kernel_init=self.kernel_init,
                         bias_init=self.bias_init, )

    # Extract Patches:
    dim = self.config.base_dim
    x = StemBlock(features=dim,
                  stem_type=self.config.stem_type,
                  **common_params)(x)

    # Network architecture is composed of n_stages (usually n_stages = 4),
    # each stage is a composition of:
    #     xN multi-head Local Self-attention blocks (LSA), followed by,
    #     xM Query-and-Attend blocks.
    # Down-sampling is performed at the last QnA-block of each stage, except for the last stage.
    drop_path_downsample = int(self.config.drop_path_downsample) * (self.config.n_stages - 1)
    drop_path = np.linspace(0.0, self.config.drop_path,
                            num=(sum(self.config.lsa_layers) + sum(self.config.qna_layers) + drop_path_downsample))
    drop_path_offset = 0
    for stage in range(self.config.n_stages):
      # Multi-head self-attention layers
      lsa_n_layers = self.config.lsa_layers[stage]
      x = LSAStage(stage=stage,
                   features=dim,
                   config=self.config,
                   drop_path=drop_path[drop_path_offset:drop_path_offset + lsa_n_layers],
                   **common_params)(x)
      drop_path_offset += lsa_n_layers
      # QnA-Layers self-attention layers
      qna_n_layers = self.config.qna_layers[stage]
      x = QnAStage(stage=stage,
                   features=dim,
                   downsample=(stage != self.config.n_stages - 1),
                   config=self.config,
                   drop_path=drop_path[
                             drop_path_offset:drop_path_offset + qna_n_layers + int(self.config.drop_path_downsample)],
                   **common_params)(x)
      drop_path_offset += qna_n_layers + int(self.config.drop_path_downsample)
      dim = x.shape[-1]

    # Get final prediction:
    if self.num_classes > 0:
      x = ClassificationHead(num_classes=self.num_classes,
                             head_type=self.config.classification_head_type,
                             num_heads=self.config.qna_heads[-1],
                             **common_params)(x)

    return x


def default_config():
  """Configurable attributes."""
  cfg = ml_collections.ConfigDict()
  cfg.n_stages = 4
  cfg.base_dim = 64

  cfg.qna_heads = [8, 16, 32, 64]
  cfg.qna_receptive_field = [3, 3, 3, 3]
  cfg.qna_layers = [1, 1, 1, 1]
  cfg.qna_pos_embedding_type = RPE_STR_REPRESENTATION
  cfg.qna_attention_scale = 'normal'
  cfg.qna_implementation = 'fused_efficient'
  cfg.qna_num_queries = 2
  cfg.qna_normalize_q = [False, False, False, False]
  cfg.qna_attention_dropout = 0.0  # TODO(marar): Add to params - currently ignored

  cfg.mlp_ratio = 4  # TODO(marar): Add to params - currently ignored
  cfg.mlp_dropout = 0.0  # TODO(marar): Add to params - currently ignored

  cfg.lsa_layers = [0, 0, 4, 2]
  cfg.lsa_window = [7, 7, 14, 7]
  cfg.lsa_heads = [2, 4, 8, 16]
  cfg.lsa_attention_dropout = 0.0
  cfg.lsa_pos_embedding_type = RPE_STR_REPRESENTATION

  cfg.activation_fn = 'gelu'  # TODO(marar): Add to params - currently ignored
  cfg.classification_head_type = 'gap'
  cfg.stem_type = 'patch'  # Or 'resnet'
  cfg.drop_path = 0.0
  cfg.layer_scale = None
  cfg.qna_normalize_stop_grads = False
  cfg.drop_path_downsample = False
  return cfg

@register
def qna_vit_tiny(config):
  cfg = default_config()

  if config.get("qna"):
    cfg.update(config.qna)

  cfg.base_dim = 64

  cfg.qna_heads = [8, 16, 32, 64]
  cfg.qna_layers = [2, 3, 2, 0]
  cfg.qna_receptive_field = [3, 3, 3, 3]

  cfg.lsa_layers = [0, 0, 4, 2]
  cfg.lsa_window = [7, 7, 14, 7]
  cfg.lsa_heads = [2, 4, 8, 16]

  cfg.drop_path = 0.05
  cfg.layer_scale = None
  cfg.qna_normalize_q = [True, True, True, True]
  cfg.qna_normalize_stop_grads = False
  cfg.drop_path_downsample = True
  return functools.partial(QnAViT, config=cfg)


@register
def qna_vit_tiny_7x7(config):
  cfg = default_config()

  if config.get("qna"):
    cfg.update(config.qna)

  cfg.base_dim = 64

  cfg.qna_heads = [8, 16, 32, 64]
  cfg.qna_layers = [2, 3, 2, 0]
  cfg.qna_receptive_field = [7, 7, 7, 7]

  cfg.lsa_layers = [0, 0, 4, 2]
  cfg.lsa_window = [7, 7, 14, 7]
  cfg.lsa_heads = [2, 4, 8, 16]

  cfg.drop_path = 0.05
  cfg.layer_scale = None
  cfg.qna_normalize_q = [True, True, True, True]
  cfg.qna_normalize_stop_grads = False
  cfg.drop_path_downsample = True
  return functools.partial(QnAViT, config=cfg)


@register
def qna_vit_small(config):
  cfg = default_config()

  if config.get("qna"):
    cfg.update(config.qna)

  cfg.base_dim = 64

  cfg.qna_heads = [8, 16, 32, 64]
  cfg.qna_layers = [2, 3, 6, 0]
  cfg.qna_receptive_field = [3, 3, 3, 3]

  cfg.lsa_layers = [0, 0, 12, 2]
  cfg.lsa_window = [7, 7, 14, 7]
  cfg.lsa_heads = [2, 4, 8, 16]

  cfg.drop_path = 0.2
  cfg.layer_scale = None
  cfg.qna_normalize_q = [True, True, True, True]
  cfg.qna_normalize_stop_grads = False
  cfg.drop_path_downsample = True
  return functools.partial(QnAViT, config=cfg)


@register
def qna_vit_base(config):
  cfg = default_config()

  if config.get("qna"):
    cfg.update(config.qna)

  cfg.base_dim = 96

  cfg.qna_heads = [6, 12, 24, 48]
  cfg.qna_layers = [2, 3, 6, 0]
  cfg.qna_receptive_field = [3, 3, 3, 3]

  cfg.lsa_layers = [0, 0, 12, 2]
  cfg.lsa_window = [7, 7, 14, 7]
  cfg.lsa_heads = [3, 6, 12, 24]

  cfg.drop_path = 0.5
  cfg.layer_scale = None
  cfg.qna_normalize_q = [True, True, True, True]
  cfg.qna_normalize_stop_grads = False
  cfg.drop_path_downsample = True
  return functools.partial(QnAViT, config=cfg)


def create_model(name, config):
  """Creates model partial function."""
  if name not in MODELS:
    raise ValueError(f"Model {name} does not exist.")
  return MODELS[name](config)
