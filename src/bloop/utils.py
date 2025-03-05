#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import operator as op
from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

__all__ = [
    "tree_dot",
    "tree_proj",
    "update_ema",
    "random_init_ema",
]


def tree_dot(tree1: Any, tree2: Any) -> float:
    return jtu.tree_reduce(op.add, jtu.tree_map(jnp.vdot, tree1, tree2))


def tree_proj(
    tree1: Any, tree2: Any, ratio_upper_bound: Optional[float], eps: float = 1e-7
) -> tuple[Any, float]:
    """Projects tree2 in the orthogonal of tree1!

    ratio gets clipped to ratio_upper_bound if it is larger than it
    in principle this ensures we are not subtracting too much
    """
    dot_product = tree_dot(tree1, tree2)
    sq_norm = tree_dot(tree1, tree1)
    ratio = dot_product / (sq_norm + eps)
    if ratio_upper_bound is not None:
        ratio = jnp.minimum(ratio, ratio_upper_bound)
    project = lambda a, b: a - ratio * b
    projected = jtu.tree_map(project, tree2, tree1)
    # returns -ratio as this is exactly the factor of g_ema in the direction
    return projected, -ratio


def update_ema(grad_main: Any, ema: float, grad_ema: Any) -> Any:
    ema_fn = lambda a, b: a + ema * (b - a)
    return jtu.tree_map(ema_fn, grad_ema, grad_main)


def random_init_ema(key: jax.Array, grad_main: Any) -> Any:
    keys = jax.random.split(key, num=len(jtu.tree_leaves(grad_main)))
    iterkeys = iter(keys)
    grad_ema = jtu.tree_map(lambda x: jax.random.normal(next(iterkeys), x.shape), grad_main)
    return grad_ema
