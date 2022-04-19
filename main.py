"""Main file for running the ImageNet example.
This file is intentionally kept short. The majority for logic is in libraries
than can be easily tested and imported in Colab.
"""

import jax
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from clu import platform
from ml_collections import config_flags

import eval
import train

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
  'config',
  None,
  'File path to the training hyperparameter configuration.',
  lock_config=True)
flags.DEFINE_bool('eval_only', False, 'Use if you want to evaluate your model')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')
  if FLAGS.eval_only:
    logging.info('========= Start Evaluation =========')
    eval.evaluate(FLAGS.config, FLAGS.workdir)
  else:
    logging.info('========= Start Training =========')
    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
