from pathlib import Path
from typing import NamedTuple

import chex
import jax
import optax
import optax.tree_utils as otu
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
from jax import random

from src.params import t0, t1, x0, x1


class MLP(nn.Module):
    widths: list[int]

    @nn.compact
    def __call__(self, x):
        for i, w in enumerate(self.widths):
            x = nn.Dense(w)(x)
            if i != len(self.widths) - 1:
                x = nn.tanh(x)
        return x


class TrainState(train_state.TrainState):
    loss: float


def create_train_state(module, key, tx):
    params = module.init(key, jnp.ones(2))["params"]
    return TrainState.create(apply_fn=module.apply, params=params, tx=tx, loss=None)


def get_rundir(model_name, cfg):
    hidden_width = cfg.module.hidden_width
    num_hidden_layers = cfg.module.num_hidden_layers
    epochs = cfg.training.epochs
    return Path(
        "runs/" + f"{model_name}/" + f"w{hidden_width}_l{num_hidden_layers}_ep{epochs}"
    )


def get_datasets(key):
    num_train_pts = 10000
    num_valid_pts = 100

    key1, key2 = random.split(key)
    train_pts = random.uniform(
        key1, (num_train_pts, 2), minval=jnp.array([t0, x0]), maxval=jnp.array([t1, x1])
    )
    valid_pts = random.uniform(
        key2, (num_valid_pts, 2), minval=jnp.array([t0, x0]), maxval=jnp.array([t1, x1])
    )

    return train_pts, valid_pts


def run_lbfgs(init_state, X, fun, opt, max_iter, tol):
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, init_state, X, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, "count")
        grad = otu.tree_get(state, "grad")
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_state.params, opt.init(init_state.params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    final_state = init_state.replace(params=final_params)
    return final_params, final_state


class InfoState(NamedTuple):
    iter_num: chex.Numeric


def print_info():
    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args

        jax.debug.print(
            "Iteration: {i}, Value: {v}, Gradient norm: {e}",
            i=state.iter_num,
            v=value,
            e=otu.tree_l2_norm(grad),
        )
        return updates, InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)
