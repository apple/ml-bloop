#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import functools

import jax
import jax.numpy as jnp
import jax.lax as lax


from loss import cross_entropy_loss, compute_metrics

from bloop.optim import (
    bloop_direction,
    dynamic_barrier_direction,
    mixed_direction,
    pcgrad_direction,
)
METHOD_DICT = {
    "mixed": mixed_direction,
    "bloop": bloop_direction,
    "pcgrad": pcgrad_direction,
    "dynamic_barrier": dynamic_barrier_direction,
}

def eval_step(state, batch):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
  return compute_metrics(logits, batch['label'])

def train_step(state, batch, learning_rate_fn):
  """Perform a single training step."""

  def loss_fn(params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch['image'],
        mutable=['batch_stats'],
    )
    loss = cross_entropy_loss(logits, batch['label'])
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_decay = 0.0001
    weight_l2 = sum(
        jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1
    )
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, (new_model_state, logits)

  step = state.step
  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')

  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr

  new_state = state.apply_gradients(
      grads=grads,
      batch_stats=lax.pmean(new_model_state['batch_stats'], 'batch'),
  )


  return new_state, metrics


def bloop_train_step(state, method_state, batch, learning_rate_fn, method):
  """Perform a single training step."""

  direction_fn = METHOD_DICT[method]

  def weight_decay_loss(params):
    weight_penalty_params = jax.tree_util.tree_leaves(params)
    weight_l2 = sum(jnp.vdot(x, x) for x in weight_penalty_params if x.ndim > 1)
    weight_penalty = 0.5 * weight_l2
    return weight_penalty
  
  def cross_entropy_loss_fn(params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        batch['image'],
        mutable=['batch_stats'],
    )
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, (new_model_state, logits)

  step = state.step
  lr = learning_rate_fn(step)

  cross_entropy_grad_fn = jax.value_and_grad(cross_entropy_loss_fn, has_aux=True)
  aux, cross_entropy_grads = cross_entropy_grad_fn(state.params)
  cross_entropy_grads = lax.pmean(cross_entropy_grads, axis_name='batch')

  weight_decay_grad_fn = jax.value_and_grad(weight_decay_loss, has_aux=False)
  weight_penalty, weight_decay_grads = weight_decay_grad_fn(state.params)
  weight_decay_grads = lax.pmean(weight_decay_grads, axis_name='batch')
    
  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr
  metrics['weight_penalty'] = weight_penalty

  grads, method_state = direction_fn(cross_entropy_grads, 
                                     weight_decay_grads, 
                                     method_state)

  new_state = state.apply_gradients(
      grads=grads,
      batch_stats=lax.pmean(new_model_state['batch_stats'], 'batch'),
  )
  return new_state, method_state, metrics