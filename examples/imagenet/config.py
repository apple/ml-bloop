#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

# This file is adapted from: https://github.com/google/flax/tree/main/examples/imagenet
#
# Please find below the original header
#
# ```
# Copyright 2024 The Flax Authors.
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
# ```
"""Default Hyperparameter configuration."""

import ml_collections

NUM_CLASSES = 1000

def default_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # As defined in the `models` module.
  config.model = 'ResNet50'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'imagenet2012:5.*.*'

  config.learning_rate = 0.1
  config.warmup_epochs = 5.0
  config.momentum = 0.9
  config.batch_size = 128
  config.shuffle_buffer_size = 16 * 128
  config.prefetch = 10

  config.num_epochs = 100.0
  config.log_every_steps = 100

  config.cache = False
  config.half_precision = False

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  # whether to profile the training loop
  config.profile = True

  return config


def metrics():
  return [
      'train_loss',
      'eval_loss',
      'train_accuracy',
      'eval_accuracy',
      'steps_per_second',
      'train_learning_rate',
  ]

def get_config():
  """Get the hyperparameter configuration to train on 8 x Nvidia V100 GPUs."""
  # Override default configuration to avoid duplication of field definition.
  config = default_config()

  config.batch_size = 128
  config.shuffle_buffer_size = 16 * 128
  config.cache = True
  config.dataset_name = "imagenet2012:5.*.*"
  config.data_dir="/root/tensorflow_datasets/"

  return config


def get_bloop_config():
  """Get the hyperparameter configuration to train on 8 x Nvidia V100 GPUs."""
  # Override default configuration to avoid duplication of field definition.
  config = get_config()
  config.bloop = True
  config.method = 'bloop'
  config.ema = 0.99
  config.lbda = 0.0001
  config.init = 'grad'
  return config
