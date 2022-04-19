import jax
from absl import logging
from jax import numpy as jnp

from models import qna_vit


def create_model(*, config, num_classes, **kwargs):
  half_precision = config.get('half_precision', False)
  logging.info(f"Creating model {config.model_name} with half_precision {half_precision}")
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32

  if config.model_name.startswith('qna'):
    model_cls = qna_vit.create_model(config.model_name, config)
  else:
    raise ValueError(f'Model {config.model_name} not supported.')

  return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)
