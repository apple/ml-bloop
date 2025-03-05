#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Any, Literal, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from flax import struct
from flax.core import frozen_dict

from bloop.utils import random_init_ema, tree_proj, update_ema

__all__ = [
    "BloopState",
    "PCGradState",
    "init_bloop",
    "init_pcgrad",
    "bloop_direction",
    "dynamic_barrier_direction",
    "pcgrad_direction",
    "mixed_direction",
]

Gradients = frozen_dict.FrozenDict[str, Any]


@struct.dataclass
class BloopState:
    grad_main_ema: Gradients
    ema: float = 0.1
    lbda: float = 0.1
    rescale: bool = struct.field(default=False, pytree_node=False)
    use_ratio_only: bool = struct.field(default=False, pytree_node=False)


@struct.dataclass
class PCGradState:
    grad_main_ema: Gradients
    grad_aux_ema: Gradients
    ema: float = 0.1
    lbda: float = 0.1
    rescale: bool = struct.field(default=False, pytree_node=False)


def init_bloop(
    key: jax.Array,
    grad_main: Gradients,
    ema: float = 0.1,
    lbda: float = 0.1,
    *,
    init: Literal["grad", "zeros", "random"] = "grad",
    rescale: bool = False,
    use_ratio_only: bool = False,
) -> BloopState:
    if init == "grad":
        grad_main_ema = jtu.tree_map(lambda x: jnp.copy(x), grad_main)
    elif init == "zeros":
        grad_main_ema = jtu.tree_map(lambda x: jnp.zeros_like(x), grad_main)
    elif init == "random":
        grad_main_ema = random_init_ema(key, grad_main)
    else:
        raise ValueError(
            f"Invalid init value {init!r}. Valid options are 'grad', 'zeros' or 'random'."
        )

    return BloopState(
        grad_main_ema,
        ema=ema,
        lbda=lbda,
        rescale=rescale,
        use_ratio_only=use_ratio_only,
    )


def init_pcgrad(
    key: jax.Array,
    grad_main: Gradients,
    ema: float = 0.1,
    lbda: float = 0.1,
    *,
    init: Literal["grad", "zeros", "random"] = "grad",
    rescale: bool = False,
) -> PCGradState:
    main_init_key, aux_init_key = jax.random.split(key, 2)
    bloop_params = init_bloop(
        main_init_key,
        grad_main,
        ema=ema,
        lbda=lbda,
        init=init,
        rescale=rescale,
    )
    if init == "random":
        grad_aux_ema = random_init_ema(aux_init_key, grad_main)
    else:
        grad_aux_ema = bloop_params.grad_main_ema

    return PCGradState(
        grad_main_ema=bloop_params.grad_main_ema,
        grad_aux_ema=grad_aux_ema,
        ema=bloop_params.ema,
        lbda=bloop_params.lbda,
        rescale=rescale,
    )


def _bloop_direction_general(
    grad_main: Gradients,
    grad_aux: Gradients,
    bloop_state: BloopState,
    ratio_upper_bound: Optional[float] = None,
    eps: float = 1e-7,
) -> tuple[Gradients, BloopState]:
    grad_main_ema = update_ema(grad_main, bloop_state.ema, bloop_state.grad_main_ema)
    lbda = bloop_state.lbda
    scale = (1.0 / 1.0 + lbda) if bloop_state.rescale else 1.0

    projected_bias, main_grad_add_factor = tree_proj(
        grad_main_ema, grad_aux, ratio_upper_bound=ratio_upper_bound, eps=eps
    )
    if bloop_state.use_ratio_only:
        direction = jtu.tree_map(
            lambda a, b: scale * ((1.0 + lbda * main_grad_add_factor) * a + lbda * b),
            grad_main,
            grad_aux,
        )
    else:
        direction = jtu.tree_map(lambda a, b: scale * (a + lbda * b), grad_main, projected_bias)

    bloop_state = bloop_state.replace(
        grad_main_ema=grad_main_ema,
    )
    return direction, bloop_state


def bloop_direction(
    grad_main: Gradients,
    grad_aux: Gradients,
    bloop_state: BloopState,
    eps: float = 1e-7,
) -> tuple[Gradients, BloopState]:
    return _bloop_direction_general(
        grad_main,
        grad_aux,
        bloop_state,
        ratio_upper_bound=None,
        eps=eps,
    )


def dynamic_barrier_direction(
    grad_main: Gradients,
    grad_aux: Gradients,
    bloop_state: BloopState,
    eps: float = 1e-7,
) -> tuple[Gradients, BloopState]:
    return _bloop_direction_general(
        grad_main,
        grad_aux,
        bloop_state,
        ratio_upper_bound=1.0 / bloop_state.lbda,
        eps=eps,
    )


def pcgrad_direction(
    grad_main: Gradients,
    grad_aux: Gradients,
    pcgrad_state: PCGradState,
    eps: float = 1e-7,
) -> tuple[Gradients, PCGradState]:
    """The original version of pcgrad!

    From https://arxiv.org/abs/2001.06782
    """
    # update EMA
    grad_main_ema = update_ema(grad_main, pcgrad_state.ema, pcgrad_state.grad_main_ema)
    grad_aux_ema = update_ema(grad_aux, pcgrad_state.ema, pcgrad_state.grad_aux_ema)
    # projecting
    ratio_upper_bound = 0
    projected_bias, train_ema_factor = tree_proj(
        grad_main_ema, grad_aux, ratio_upper_bound=ratio_upper_bound, eps=eps
    )
    projected_train, _ = tree_proj(
        grad_aux_ema, grad_main, ratio_upper_bound=ratio_upper_bound, eps=eps
    )
    # compute direction
    lbda = pcgrad_state.lbda
    scale = 1.0 / (1.0 + lbda) if pcgrad_state.rescale else 1.0

    direction = jtu.tree_map(lambda a, b: scale * (a + lbda * b), projected_train, projected_bias)
    pcgrad_state = pcgrad_state.replace(
        grad_main_ema=grad_main_ema,
        grad_aux_ema=grad_aux_ema,
    )
    return direction, pcgrad_state


def mixed_direction(
    grad_main: Gradients,
    grad_aux: Gradients,
    bloop_state: BloopState,
) -> tuple[Gradients, BloopState]:
    lbda = bloop_state.lbda
    scale = 1.0 / (1.0 + lbda) if bloop_state.rescale else 1.0

    direction = jtu.tree_map(lambda a, b: scale * (a + lbda * b), grad_main, grad_aux)
    return direction, bloop_state
