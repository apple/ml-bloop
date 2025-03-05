#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import pytest

import jax
import jax.numpy as jnp

from bloop.optim import (
    bloop_direction,
    dynamic_barrier_direction,
    init_bloop,
    init_pcgrad,
    mixed_direction,
    pcgrad_direction,
    update_ema,
)


@pytest.mark.parametrize("init", ["grad", "zeros", "random"])
@pytest.mark.parametrize("rescale", [True, False])
@pytest.mark.parametrize("use_ratio_only", [True, False])
def test_init(init, rescale, use_ratio_only):
    lbda = 0.1
    ema = 0.1
    grad_main = jnp.ones((3, 2))
    key = jax.random.PRNGKey(0)
    _ = init_bloop(
        key,
        grad_main,
        lbda=lbda,
        ema=ema,
        init=init,
        rescale=rescale,
        use_ratio_only=use_ratio_only,
    )
    _ = init_pcgrad(
        key,
        grad_main,
        lbda=lbda,
        ema=ema,
        init=init,
        rescale=rescale,
    )


@pytest.mark.parametrize("method", ["bloop", "mixed", "dynamic"])
def test_directions(method):
    lbda = 0.1
    ema = 0.1
    direction_fn = {
        "bloop": bloop_direction,
        "mixed": mixed_direction,
        "dynamic": dynamic_barrier_direction,
    }[method]

    shape = (3, 4)
    grad_main = jnp.ones(shape)
    grad_aux = jnp.zeros(shape)
    key = jax.random.PRNGKey(0)
    bloop_state = init_bloop(key, grad_main, lbda=lbda, ema=ema)
    direction, bloop_state = jax.jit(direction_fn)(grad_main, grad_aux, bloop_state)
    assert direction.shape == shape


def test_pcgrad():
    lbda = 0.1
    ema = 0.1
    shape = (3, 4)
    grad_main = jnp.ones(shape)
    grad_aux = jnp.zeros(shape)
    key = jax.random.PRNGKey(0)
    pcgrad_state = init_pcgrad(key, grad_main, lbda=lbda, ema=ema)
    direction, pcgrad_state = jax.jit(
        pcgrad_direction,
    )(grad_main, grad_aux, pcgrad_state)
    assert direction.shape == shape


@pytest.mark.parametrize("ema", [1.0, 0.0])
def test_ema(ema):
    rng = jax.random.PRNGKey(0)
    shape = (3, 4)
    grad_main = jax.random.normal(rng, shape)
    rng, _ = jax.random.split(rng)
    grad_ema = jax.random.normal(rng, shape)
    grad_ema_new = update_ema(grad_main, ema, grad_ema)
    if ema == 0.0:
        jnp.allclose(grad_ema_new, grad_ema)
    if ema == 1.0:
        assert jnp.allclose(grad_ema_new, grad_main)
