# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Based on : https://github.com/google/flax/blob/main/examples/imagenet/train.py"""
import functools
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
import tensorflow as tf
from absl import logging
from flax import jax_utils
from flax.core import frozen_dict
from flax.training import checkpoints
from flax.training import common_utils
from jax import lax
from jax import random

import models
from data import input_pipeline


def cross_entropy_loss(logits, labels, num_classes=10):
  if len(labels.shape) == 1:
    # Convert one-hot labels to single values if appliable.
    one_hot_labels = common_utils.onehot(labels, num_classes=num_classes)
  else:
    one_hot_labels = labels
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels, num_classes=10):
  loss = cross_entropy_loss(logits, labels, num_classes)
  if len(labels.shape) == 1:
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  else:
    accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  metrics = {
    'loss': loss,
    'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    if len(x.shape) == 4:
      return x.reshape((local_device_count, -1) + x.shape[1:])
    else:
      return x

  return jax.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, config, rng, batch_size, train, image_size, dtype, cache):
  ds = input_pipeline.create_split(
    dataset_builder,
    config,
    rng,
    batch_size,
    train=train,
    image_size=image_size,
    dtype=dtype,
    cache=cache,
    drop_remainder=False,
    num_epochs=1)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 10)
  return it


def get_model_params(workdir):
  restored = checkpoints.restore_checkpoint(workdir, target=None)
  if 'params' not in restored.keys():
    raise ValueError('Checkpoint corrupted - model params not available')

  params = frozen_dict.freeze(restored['params'])
  batch_stats = None
  if 'batch_stats' in restored.keys():
    batch_stats = restored['batch_stats']

  step = -1
  if 'step' in restored.keys():
    step = restored['step']
  return params, batch_stats, step


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def _get_data_iter(config: ml_collections.ConfigDict, rng: Any):
  image_size = config.input_size

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.local_device_count()

  if config.half_precision:
    platform = jax.local_devices()[0].platform
    if platform == 'tpu':

      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
  else:
    input_dtype = tf.float32

  # Build input pipeline.
  dataset_builder = input_pipeline.get_dataset_builder(config)

  # TODO(marar): data_rng_val is not necessary!
  eval_iter = create_input_iter(
    dataset_builder, config, rng, batch_size=local_batch_size, train=False, image_size=image_size,
    dtype=input_dtype,
    cache=config.cache)

  num_classes = dataset_builder.info.features['label'].num_classes
  return eval_iter, num_classes


def eval_step(batch, params, batch_stats, model_fn, num_classes):
  variables = {'params': params}
  if batch_stats:
    variables['batch_stats'] = batch_stats
  logits = model_fn(variables, batch['image'], train=False, mutable=False)
  labels = batch['label']
  # loss = cross_entropy_loss(logits, labels, num_classes)
  if len(labels.shape) == 1:
    correct = jnp.sum(jnp.argmax(logits, -1) == labels)
  else:
    correct = jnp.sum(jnp.argmax(logits, -1) == jnp.argmax(labels, -1))
  return correct


def evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Perfirm model evaluation.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  rng = random.PRNGKey(config.seed)
  rng, data_rng = jax.random.split(rng)
  logging.info('=== Creating data iterator')
  eval_iter, num_classes = _get_data_iter(config, data_rng)
  logging.info('=== Creating model')
  model = models.create_model(config=config, num_classes=num_classes)
  logging.info('=== Get model params')
  model_params, model_batch_stats, step = get_model_params(workdir)
  logging.info('=== Create evaluation function')
  # Create evaluation step fn:
  num_classes = input_pipeline.get_num_classes_from_config(config)
  static_params = {'model_fn': model.apply,
                   'num_classes': num_classes,
                   }
  model_params = jax_utils.replicate(model_params)
  if model_batch_stats:
    model_batch_stats = jax_utils.replicate(model_batch_stats)
  # else:
  #    static_params['batch_stats'] = None
  p_eval_step = jax.pmap(functools.partial(eval_step, **static_params), axis_name='batch')
  first_call = True
  logging.info('=== Start evaluation:')
  total_correct = 0
  total_count = 0
  for batch in eval_iter:
    if first_call:  # The first call to the model once will compile the computation graph
      logging.info('Initial compilation - this may take a while')
      first_call = False
    correct = p_eval_step(batch, model_params, model_batch_stats)
    total_correct += jnp.sum(correct)
    total_count += batch['image'].shape[0] * batch['image'].shape[1]
  logging.info('========== Model Accuracy : %.2f', (total_correct / total_count) * 100)
