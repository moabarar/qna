from typing import Dict, Union

import tensorflow as tf

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2471, 0.2435, 0.2616]
CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Features = Dict[str, Tensor]


def normalize_image(image, mean_rgb, std_rgb):
  image -= tf.constant(mean_rgb, shape=[1, 1, 3], dtype=image.dtype)
  image /= tf.constant(std_rgb, shape=[1, 1, 3], dtype=image.dtype)
  return image


def normalize_image_cifar10(image):
  return normalize_image(image, CIFAR10_MEAN, CIFAR10_STD)


def normalize_image_cifar100(image):
  return normalize_image(image, CIFAR100_MEAN, CIFAR100_STD)


# TODO(marar): remove
def decode_and_random_resized_crop(image: tf.Tensor, rng,
                                   resize_size: int) -> tf.Tensor:
  """Decodes the images and extracts a random crop."""
  shape = tf.io.extract_jpeg_shape(image)
  begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
    shape,
    tf.zeros([0, 0, 4], tf.float32),
    seed=rng,
    area_range=(0.05, 1.0),
    min_object_covered=0,  # Don't enforce a minimum area.
    use_image_if_no_bounding_boxes=True)
  top, left, _ = tf.unstack(begin)
  h, w, _ = tf.unstack(size)
  image = tf.image.decode_and_crop_jpeg(image, [top, left, h, w], channels=3)
  # image = tf.cast(image, tf.float32) / 255.0
  image = tf.image.resize(image, (resize_size, resize_size))
  return image


def preprocess_for_train(image, rng, dtype, **kwargs):
  """Augmentation function for cifar dataset."""
  _, rng_crop, rng_flip = tf.unstack(
    tf.random.experimental.stateless_split(rng, 3))
  image = tf.io.decode_jpeg(image)
  image = tf.image.resize_with_crop_or_pad(image, 32 + 4, 32 + 4)
  image = tf.image.stateless_random_crop(image, [32, 32, 3], seed=rng_crop)
  image = tf.image.stateless_random_flip_left_right(image, seed=rng_flip)
  image = tf.cast(image, dtype)
  return image


def preprocess_for_eval(image, rng, dtype, **kwargs):
  """Processes a single example for evaluation for cifar."""
  image = tf.io.decode_jpeg(image)
  image = tf.cast(image, dtype)
  return image
