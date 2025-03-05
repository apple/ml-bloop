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
"""ImageNet example.

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import time
from typing import Any
import logging
import tqdm
import ml_collections

from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import optax



import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.experimental.set_visible_devices([], 'GPU')

import input_pipeline
import models
from config import NUM_CLASSES

import train_step as step_lib
from bloop.optim import init_bloop, init_pcgrad

def create_model(*, model_cls, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(num_classes=NUM_CLASSES, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)

  @jax.jit
  def init(*args):
    return model.init(*args)

  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  return variables['params'], variables['batch_stats']



def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int,
):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0.0,
      end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch,
  )
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
  )
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch],
  )
  return schedule_fn


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()

  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_util.tree_map(_prepare, xs)


def create_input_iter(
    dataset_name,
    data_dir,
    batch_size,
    image_size,
    dtype,
    train,
    cache,
    shuffle_buffer_size,
    prefetch,
):
  ds = input_pipeline.create_split(
      dataset_name,
      data_dir,
      batch_size,
      image_size=image_size,
      dtype=dtype,
      train=train,
      cache=cache,
      shuffle_buffer_size=shuffle_buffer_size,
      prefetch=prefetch,
  )
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


class TrainState(train_state.TrainState):
  batch_stats: Any


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  step = int(state.step)
  logging.info('Saving checkpoint step %d.', step)
  checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
  """Create initial training state."""
  platform = jax.local_devices()[0].platform

  params, batch_stats = initialized(rng, image_size, model)
  tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=True,
  )
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      batch_stats=batch_stats
  )
  return state


def create_method_state(rng, config, state):
    """Creates initial `MethodState`."""
    if config.method in ["mixed", "bloop", "dynamic_barrier"]:
        method_state = init_bloop(
            rng, state.params, ema=config.ema, lbda=config.lbda, init=config.init
        )
    elif config.method == "pcgrad":
        method_state = init_pcgrad(
            rng, state.params, ema=config.ema, lbda=config.lbda, init=config.init
        )
    else:
        raise ValueError(f"Unknown method: {config.method}.")
    return method_state

def train_and_evaluate(
    config: ml_collections.ConfigDict,
    workdir: str, 
    log_func: Any=None,
) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  )

  rng = random.key(0)

  image_size = 224

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()

  platform = jax.local_devices()[0].platform

  if config.half_precision:
    if platform == 'tpu':
      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
  else:
    input_dtype = tf.float32

  dataset_builder = tfds.builder(config.dataset)
  train_iter = create_input_iter(
      config.dataset_name,
      config.data_dir,
      local_batch_size,
      image_size,
      input_dtype,
      train=True,
      cache=config.cache,
      shuffle_buffer_size=config.shuffle_buffer_size,
      prefetch=config.prefetch,
  )
  eval_iter = create_input_iter(
      config.dataset_name,
      config.data_dir,
      local_batch_size,
      image_size,
      input_dtype,
      train=False,
      cache=config.cache,
      shuffle_buffer_size=None,
      prefetch=config.prefetch,
  )

  steps_per_epoch = (
      dataset_builder.info.splits['train'].num_examples // config.batch_size
  )

  if config.num_train_steps <= 0:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        'validation'
    ].num_examples
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  steps_per_checkpoint = steps_per_epoch * 10

  base_learning_rate = config.learning_rate * config.batch_size / 256.0

  model_cls = getattr(models, config.model)
  model = create_model(
      model_cls=model_cls, half_precision=config.half_precision
  )

  learning_rate_fn = create_learning_rate_fn(
      config, base_learning_rate, steps_per_epoch
  )
  
  rng, init_state_rng, method_rng = jax.random.split(rng, 3)
  state = create_train_state(init_state_rng, config, model, image_size, learning_rate_fn)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  
  if hasattr(config, 'bloop'):
    method_state = create_method_state(method_rng, config, state)
    p_train_step = jax.pmap(
        functools.partial(step_lib.bloop_train_step, 
                          method = config.method,
                          learning_rate_fn=learning_rate_fn),
        in_axes=(None, None, 0),
        out_axes=(None, None, 0),
        axis_name='batch',
    )
  else:
    p_train_step = jax.pmap(
        functools.partial(step_lib.train_step, learning_rate_fn=learning_rate_fn),
        in_axes=(None, 0),
        out_axes=(None, 0),
        axis_name='batch',
    )
  p_eval_step = jax.pmap(step_lib.eval_step, in_axes=(None, 0), axis_name='batch')

  train_metrics = []
  hooks = []
  if jax.process_index() == 0 and config.profile:
    hooks += [
        periodic_actions.Profile(
            num_profile_steps=3, profile_duration_ms=None, logdir=workdir
        )
    ]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for step, batch in tqdm.tqdm(zip(range(step_offset, num_steps), train_iter)):
    if hasattr(config, 'bloop'):
      state, method_state, metrics = p_train_step(state, method_state, batch)
    else:
      state, metrics = p_train_step(state, batch)
    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
        train_metrics = common_utils.get_metrics(train_metrics)
        summary = {
            f'train_{k}': v
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), train_metrics
            ).items()
        }
        summary['steps_per_second'] = config.log_every_steps / (
            time.time() - train_metrics_last_t
        )
        if log_func is not None:
          log_func(summary)
        writer.write_scalars(step + 1, summary)
        train_metrics = []
        train_metrics_last_t = time.time()

    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      eval_metrics = []

      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      # summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
      summary = {
            f'eval_{k}': v
            for k, v in jax.tree_util.tree_map(
                lambda x: x.mean(), eval_metrics
            ).items()
        }
      logging.info(
          'eval epoch: %d, loss: %.4f, accuracy: %.2f',
          epoch,
          summary['eval_loss'],
          summary['eval_accuracy'] * 100,
      )

      if log_func is not None:
          log_func(summary)
      writer.write_scalars(step + 1, summary)
      writer.flush()
    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()

  return state

if __name__ == "__main__":
  import config as config_lib 
  import sys
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-b", "--use_bloop", help="use_bloop",
                  action="store_true")
  args = parser.parse_args()
  use_bloop =  args.use_bloop


  if use_bloop:
    config = config_lib.get_bloop_config()
  else:
    config = config_lib.get_config()
  logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
  train_and_evaluate(config=config, workdir='./')