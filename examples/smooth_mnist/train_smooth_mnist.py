#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

# This file is adapted from: https://github.com/google/flax/blob/main/examples/mnist/train.py
#
# Please find below the original header
#
# ```
# Copyright 2023 The Flax Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ```

"""MNIST example.

Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# See issue #620.
# pytype: disable=wrong-keyword-args

import argparse
import os
from collections.abc import Sequence
from copy import deepcopy
from time import time
from typing import Any

import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from objprint import op

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state

from bloop.optim import (
    bloop_direction,
    dynamic_barrier_direction,
    init_bloop,
    init_pcgrad,
    mixed_direction,
    pcgrad_direction,
)


class MLP(nn.Module):
    """A simple MLP model for MNIST."""

    num_classes: int = 10
    n_features: Sequence[int] = (256, 128)
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, train=True):
        x = x.reshape((x.shape[0], -1))
        for n in self.n_features:
            x = nn.Dense(n, dtype=self.dtype)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        return x


def main_value_and_grad(state, images, labels):
    """Computes gradients, loss and accuracy for the main loss."""

    def loss_fn(params):
        """The main loss function is cross-entropy."""
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, grads, accuracy


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    mean = 0.1307
    std = 0.3081
    train_ds["image"] = jnp.float32(train_ds["image"]) / 255.0
    train_ds["image"] = (train_ds["image"] - mean) / std
    test_ds["image"] = jnp.float32(test_ds["image"]) / 255.0
    test_ds["image"] = (test_ds["image"] - mean) / std
    return train_ds, test_ds


def lipschitz_loss_fn(params, *, square=True, log=True):
    """The auxiliary loss ensures the obtained network is smooth.

    It computes the upper bound on the Lipschitz constant for an MLP.
    """

    flat_params = jax.tree_util.tree_leaves(params)
    loss_val = 1.0
    for x in flat_params:
        if x.ndim > 1:
            norm = jnp.linalg.norm(x, ord=2)
            if square:
                norm = norm**2
            loss_val = loss_val * norm
    if log:
        loss_val = jnp.log(loss_val)
    return loss_val


def aux_value_and_grad(state):
    """Computes gradients and loss value for the auxiliary loss."""
    grad_fn = jax.value_and_grad(lipschitz_loss_fn)
    return grad_fn(state.params)


def update_model(state, grads):
    return state.apply_gradients(grads=grads)


METHOD_DICT = {
    "mixed": mixed_direction,
    "bloop": bloop_direction,
    "pcgrad": pcgrad_direction,
    "dynamic_barrier": dynamic_barrier_direction,
}


def train_epoch(state, method_state, train_ds, batch_size, method, rng):
    """Train for a single epoch."""

    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds["image"]))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_loss = []
    epoch_accuracy = []
    direction_fn = METHOD_DICT[method]

    def update_function(state, method_state, batch_images, batch_labels):
        # Get gradients from the two losses
        main_loss, main_gradient, accuracy = main_value_and_grad(state, batch_images, batch_labels)
        aux_loss, aux_gradient = aux_value_and_grad(state)
        # Apply gradient surgery
        direction, method_state = direction_fn(main_gradient, aux_gradient, method_state)
        # Update model
        state = update_model(state, direction)
        return state, method_state, main_loss, accuracy

    update_function = jax.jit(update_function, donate_argnums=(0, 1))

    for perm in perms:
        # Get data
        batch_images = train_ds["image"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        state, method_state, main_loss, accuracy = update_function(
            state, method_state, batch_images, batch_labels
        )
        epoch_loss.append(main_loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, method_state, train_loss, train_accuracy


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    model = MLP()
    params = model.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
    # Use Adam optimizer
    tx = optax.adam(config.learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


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


def train_and_evaluate(config: argparse.Namespace) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.

    Returns:
      - The train state (which includes the `.params`).
      - The evaluation metrics.
    """

    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)

    # Metrics to track
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    aux_losses = []

    # Create training and method states
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)
    method_state = create_method_state(rng, config, state)

    # Main training loop
    for epoch in range(1, config.num_epochs + 1):
        t0 = time()
        rng, input_rng = jax.random.split(rng)
        state, method_state, train_loss, train_accuracy = train_epoch(
            state,
            method_state,
            train_ds,
            config.batch_size,
            config.method,
            input_rng,
        )
        test_loss, _, test_accuracy = main_value_and_grad(state, test_ds["image"], test_ds["label"])
        aux_loss = lipschitz_loss_fn(state.params)
        print(
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f,"
            " test_accuracy: %.2f, Lipschitz upper bound: %.2f, time:%.2f"
            % (
                epoch,
                train_loss,
                train_accuracy * 100,
                test_loss,
                test_accuracy * 100,
                aux_loss,
                time() - t0,
            )
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        aux_losses.append(aux_loss)

    metrics = {
        "train_loss": train_losses,
        "train_accuracy": train_accuracies,
        "test_loss": test_losses,
        "test_accuracy": test_accuracies,
        "aux_loss": aux_losses,
    }
    return state, metrics


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training MNIST with lipschitz regularization")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["mixed", "bloop", "pcgrad", "dynamic_barrier"],
        choices=["mixed", "bloop", "pcgrad", "dynamic_barrier"],
        help="List of methods to use.",
    )
    parser.add_argument(
        "--emas",
        type=float,
        nargs="+",
        default=[1, 1e-2, 1, 1],
        help="List of the corresponding emas.",
    )
    parser.add_argument(
        "--lbdas",
        type=float,
        nargs="+",
        default=None,
        help="List of lbdas to train the model with.",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=25, help="Number of epochs.")
    parser.add_argument(
        "--init",
        type=str,
        default="grad",
        choices=["random", "zeros", "grad"],
    )
    parser.add_argument(
        "--metric_dir", type=str, default="results/metrics", help="Directory to save metrics."
    )

    args = parser.parse_args()
    os.makedirs(args.metric_dir, exist_ok=True)
    # Avoid tensorflow from using the GPU
    tf.config.experimental.set_visible_devices([], "GPU")

    if args.lbdas is None:
        lbdas = jnp.logspace(-4, 0, 9)
    else:
        lbdas = args.lbdas
        del args.lbdas

    configs = []
    for lbda in lbdas:
        for method, ema in zip(args.methods, args.emas):
            config = deepcopy(args)
            del config.methods
            del config.emas
            config.lbda = lbda
            config.ema = ema
            config.method = method
            configs.append(config)

    for config in configs:
        op(vars(config))
        filename = f"lbda_{config.lbda:.5f}_ema_{config.ema}_method_{config.method}".replace(
            ".", "-"
        )
        savestr = os.path.join(args.metric_dir, filename)
        if os.path.exists(savestr + ".npz"):
            continue
        state, metrics = train_and_evaluate(config)
        np.savez(savestr, **metrics)
