"""A config for training QnA on ImageNet."""

import ml_collections


def get_config():
  """Configs."""
  config = ml_collections.ConfigDict()

  # General configs
  config.model_name = "qna_vit_base"
  config.batch_size = 1024
  config.seed = 0
  config.trial = 0
  config.half_precision = True
  config.fine_tune = True
  config.pretrained_path = ''

  # Add training hyperparams:
  config.num_epochs = 30
  config.num_train_steps = -1
  config.steps_per_eval = -1
  config.steps_per_eval = -1
  config.eval_per_epochs = 5
  config.checkpoint_every_epochs = 5
  config.log_every_steps = 500

  # Setup the image size by changing this CONST
  IMG_SIZE = 384

  # Add Data related configs.
  config.input_size = IMG_SIZE
  config.cache = True
  config.dataset = "imagenet2012"
  config.dataset_version = "5.0.0"
  config.data_dir = 'none'
  config.shuffle_buffer_size = 10000

  # Add training related configs:
  config.learning_rate = 1e-5
  config.optim = "adamw"
  config.optim_wd_ignore = [
    'attn_scale_weights',
    'attn_scale_weights_pos',
    'attn_scale_weights_neg',
    'rpe_bias',
    'rpe_bias_qc',
    'rpe_bias_qr'
    'pos_embedding',
    'query',
    'cls_token',
    'pos_scale',
    'neg_scale'
  ]
  config.grad_clip_max_norm = False
  config.learning_rate_schedule = "const"
  config.warmup_epochs = 0
  config.weight_decay = 1e-8

  # Augmentations:
  config.randaugment = ml_collections.ConfigDict()
  config.randaugment.type = "randaugment"  # Set to `default` to disable
  # All parameters start with `config.augment.randaugment_`.
  config.randaugment.randaugment_num_layers = 2
  config.randaugment.randaugment_cutout = False
  config.randaugment.randaugment_magnitude = 9
  config.randaugment.randaugment_magstd = 0.5
  config.randaugment.randaugment_prob_to_apply = 0.5
  config.randaugment.size = IMG_SIZE
  # Add random erasing.
  config.randerasing = ml_collections.ConfigDict()
  config.randerasing.erase_prob = 0.25  # Set to 0 to disable
  # Add mix style augmentation.
  config.mix = ml_collections.ConfigDict()
  config.mix.mixup_alpha = 0.8
  config.mix.prob_to_apply = 1.0  # Set to 0 to disable
  config.mix.smoothing = 0.1

  # Add color jitter.
  # It uses default impl='simclrv2'
  config.colorjitter = ml_collections.ConfigDict()
  config.colorjitter.type = "colorjitter"  # Set to `default` to disable
  config.colorjitter.colorjitter_strength = 0.3
  config.colorjitter.size = IMG_SIZE
  return config
