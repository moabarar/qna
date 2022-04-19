"""Data Utils."""

from typing import Any, Callable

import jax.numpy as jnp
import jax.random
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
from clu import deterministic_data

import data.cifar_utils as cifar_utils
import data.imagenet_utils as imagenet_utils
from augment import augment_utils

Features = Any


def _get_base_preprocess_fn(dataset_name, train):
  if train:
    if dataset_name.startswith('cifar'):
      preprocess_fn = cifar_utils.preprocess_for_train
    else:
      assert dataset_name.startswith('imagenet')
      preprocess_fn = imagenet_utils.preprocess_for_train
  else:
    if dataset_name.startswith('cifar'):
      preprocess_fn = cifar_utils.preprocess_for_eval
    else:
      assert dataset_name.startswith('imagenet')
      preprocess_fn = imagenet_utils.preprocess_for_eval
  return preprocess_fn


def _get_normalization_fn(dataset_name):
  if dataset_name == 'cifar10':
    return cifar_utils.normalize_image_cifar10
  elif dataset_name == 'cifar100':
    return cifar_utils.normalize_image_cifar100
  elif dataset_name.startswith('imagenet'):
    return imagenet_utils.normalize_image
  else:
    raise ValueError(f'Dataset {dataset_name} is not supported')


def preprocess_with_per_batch_rng(ds: tf.data.Dataset,
                                  preprocess_fn: Callable[[Features], Features],
                                  *, rng: jnp.ndarray) -> tf.data.Dataset:
  """Maps batched `ds` using the preprocess_fn and a deterministic RNG per batch.

  This preprocess_fn usually contains data preprcess needs a batch of data, like
  Mixup.

  Args:
    ds: Dataset containing Python dictionary with the features. The 'rng'
      feature should not exist.
    preprocess_fn: Preprocessing function that takes a Python dictionary of
      tensors and returns a Python dictionary of tensors. The function should be
      convertible into a TF graph.
    rng: Base RNG to use. Per example RNGs will be derived from this by folding
      in the example index.

  Returns:
    The dataset mapped by the `preprocess_fn`.
  """
  rng = list(jax.random.split(rng, 1)).pop()

  def _fn(example_index: int, features: Features) -> Features:
    example_index = tf.cast(example_index, tf.int32)
    features["rng"] = tf.random.experimental.stateless_fold_in(
      tf.cast(rng, tf.int64), example_index)
    processed = preprocess_fn(features)
    if isinstance(processed, dict) and "rng" in processed:
      del processed["rng"]
    return processed

  return ds.enumerate().map(
    _fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def create_split_old(dataset_builder: tfds.core.DatasetBuilder,
                     config: ml_collections.ConfigDict,
                     data_rng: Any,
                     batch_size: int, train: bool, image_size: int,
                     dtype=tf.float32,
                     cache=False, prefetch=10):
  """Creates a split from the ImageNet dataset using TensorFlow Datasets.

  Args:
    dataset_builder: TFDS dataset builder for ImageNet/Cifar10/Cifar100.
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    image_size: The target size of the images.
    dtype: data type of the image.
    cache: Whether to cache the dataset.
    prefetch: The number of prefetched batches
  Returns:
    A `tf.data.Dataset`.
  """
  dataset = dataset_builder.name
  if train:
    split = deterministic_data.get_read_instruction_for_host("train", dataset_info=dataset_builder.info)
    if dataset.startswith('cifar'):
      preprocess_fn = cifar_utils.preprocess_for_train
    else:
      assert dataset.startswith('imagenet')
      preprocess_fn = imagenet_utils.preprocess_for_train
  else:
    validation_key = ('validation' if dataset.startswith('imagenet') else 'testing')
    split = deterministic_data.get_read_instruction_for_host(validation_key, dataset_info=dataset_builder.info)
    if dataset.startswith('cifar'):
      preprocess_fn = cifar_utils.preprocess_for_eval
    else:
      assert dataset.startswith('imagenet')
      preprocess_fn = imagenet_utils.preprocess_for_eval

  def decode_example(example):
    image = preprocess_fn(example['image'], image_size=image_size, rng=data_rng)
    return {'image': image, 'label': example['label']}

  ds = dataset_builder.as_dataset(split=split, decoders={
    'image': tfds.decode.SkipDecoding(),
  })
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 48
  ds = ds.with_options(options)

  if cache:
    ds = ds.cache()

  if train:
    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=0)

  ds = ds.map(decode_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  ds = ds.batch(batch_size, drop_remainder=True)

  if not train:
    ds = ds.repeat()

  ds = ds.prefetch(prefetch)

  return ds


def _get_preprocess_fn(train: bool, dataset_name: str, randaugment_params=None, colorjitter_params=None,
                       randerasing_params=None):
  # Define basic preprocessing function
  preprocess_fn = _get_base_preprocess_fn(dataset_name, train)
  normalization_fn = _get_normalization_fn(dataset_name)

  # Setup augmentations for train only!
  rand_augmentor = None
  colorjitter_augmentor = None
  randerasing_augmentor = None
  if train:
    if randaugment_params is not None:
      rand_augmentor = augment_utils.create_augmenter(**randaugment_params.to_dict())
    if colorjitter_params is not None:
      colorjitter_augmentor = augment_utils.create_augmenter(**colorjitter_params.to_dict())
    if randerasing_params is not None:
      randerasing_augmentor = augment_utils.create_random_erasing(**randerasing_params.to_dict())

  def preprocess_combined(image, rng, dtype, image_size):
    rng_basic, rng_ra, rng_jt, rng_re = tf.unstack(
      tf.random.experimental.stateless_split(rng, 4))
    # Return image as uint8 since RandAugment and Colorjitter work on uint8 images
    image = preprocess_fn(image, image_size=image_size, rng=rng_basic, dtype=tf.uint8)
    if rand_augmentor is not None:
      image = rand_augmentor(rng_ra, image)
    if colorjitter_augmentor is not None:
      image = colorjitter_augmentor(rng_jt, image)
    # Cast image to output dtype
    image = tf.cast(image, dtype)
    image = normalization_fn(image)
    if randerasing_augmentor is not None:
      image = randerasing_augmentor(rng_re, image)
    return image

  return preprocess_combined


def _get_postprocess_fn(config, num_classes):
  batch_preprocess_fn = None
  if config.get('mix', None) is not None:
    batch_preprocess_fn = augment_utils.create_mix_augment(
      num_classes=num_classes,
      **config.mix.to_dict())
  return batch_preprocess_fn


def create_split(dataset_builder: tfds.core.DatasetBuilder,
                 config: ml_collections.ConfigDict,
                 data_rng: Any,
                 batch_size: int, train: bool, image_size: int,
                 dtype=tf.float32,
                 cache=False, prefetch=10):
  """Creates a split from the dataset using TensorFlow Datasets.

  Args:
    dataset_builder: TFDS dataset builder for ImageNet/Cifar10/Cifar100.
    batch_size: the batch size returned by the data pipeline.
    train: Whether to load the train or evaluation split.
    image_size: The target size of the images.
    dtype: data type of the image.
    cache: Whether to cache the dataset.
    prefetch: The number of prefetched batches
  Returns:
    A `tf.data.Dataset`.
  """
  dataset = dataset_builder.name
  if train:
    split = deterministic_data.get_read_instruction_for_host("train", dataset_info=dataset_builder.info)
  else:
    validation_key = ('validation' if dataset.startswith('imagenet') else 'test')
    split = deterministic_data.get_read_instruction_for_host(validation_key, dataset_info=dataset_builder.info)
  preprocess_fn = _get_preprocess_fn(train, dataset, randaugment_params=config.get('randaugment', None),
                                     colorjitter_params=config.get('colorjitter', None),
                                     randerasing_params=config.get('randerasing'))

  postprocess_fn = (
    _get_postprocess_fn(config, num_classes=dataset_builder.info.features['label'].num_classes)
    if train else None)

  def decode_example(example):
    rng = example.get('rng', None)
    image = preprocess_fn(example['image'], rng=rng, dtype=dtype, image_size=image_size)
    del example['rng']
    return {'image': image, 'label': example['label']}

  data_rng1, data_rng2 = jax.random.split(data_rng, 2)

  batch_dims = [jax.local_device_count(), batch_size]
  ds = deterministic_data.create_dataset(
    dataset_builder,
    split=split,
    rng=data_rng1,
    preprocess_fn=decode_example,
    cache=cache,
    decoders={"image": tfds.decode.SkipDecoding()},
    shuffle_buffer_size=jax.local_device_count() * batch_size,
    batch_dims=batch_dims if postprocess_fn is None else None,
    num_epochs=None,
    shuffle=True,
    prefetch_size=prefetch,
    drop_remainder=True,
  )
  if postprocess_fn is not None:
    ds = ds.batch(batch_dims[-1], drop_remainder=True)
    ds = preprocess_with_per_batch_rng(
      ds, postprocess_fn, rng=data_rng2)
    for batch_size in reversed(batch_dims[:-1]):
      ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(prefetch)
  return ds


def get_dataset_builder(config):
  data_dir = (
    None if (config.get('data_dir', None) == 'none' or config.get('data_dir', None) is None) else config.data_dir)
  if config.dataset.startswith('imagenet'):
    dataset_builder = tfds.builder(config.dataset, data_dir=data_dir, version=config.dataset_version)
    dataset_builder.download_and_prepare()
  else:
    dataset_builder = tfds.builder(config.dataset, data_dir=data_dir)
    dataset_builder.download_and_prepare()
  return dataset_builder


def get_num_eval_examples(dataset_builder, config):
  if config.dataset.startswith('imagenet'):
    return dataset_builder.info.splits[
      'validation'].num_examples
  elif config.dataset.startswith('cifar'):
    return dataset_builder.info.splits[
      'testing'].num_examples
  else:
    raise ValueError(f'Dataset {config.dataset} not supported')


def get_num_classes_from_config(config):
  if config.dataset == 'cifar10':
    return 10
  elif config.dataset == 'cifar100':
    return 100
  elif config.dataset == 'imagenet2012':
    return 1000
