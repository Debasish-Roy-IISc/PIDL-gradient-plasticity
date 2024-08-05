import argparse
import pickle
import shutil
import sys
import time
import tomllib
from pathlib import Path

import jax
import munch
import optax
import optax.tree_utils as otu
import pandas as pd
import tomli_w
from flax import serialization
from jax import numpy as jnp
from jax import random
from matplotlib import pyplot as plt

from src import common
from src.models import (chap2_le0_ld0_H0, chap2_le0_ld0_H500,
                        chap2_le0_ld10_H0, chap2_le0_ld10_H500,
                        chap2_le10_ld0_H0, chap2_le10_ld0_H500,
                        chap2_le10_ld5_H200, chap2_le0_ld10_H0_mixed)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Model name")
parser.add_argument("--mode", choices=["train", "test"])
parser.add_argument("--config", help="Config file path")
parser.add_argument("--rundir")
args = parser.parse_args()


def main():
    # Select the model
    match args.model:
        case "chap2_le0_ld0_H0":
            model = chap2_le0_ld0_H0
        case "chap2_le0_ld0_H500":
            model = chap2_le0_ld0_H500
        case "chap2_le10_ld0_H0":
            model = chap2_le10_ld0_H0
        case "chap2_le10_ld0_H500":
            model = chap2_le10_ld0_H500
        case "chap2_le0_ld10_H0":
            model = chap2_le0_ld10_H0
        case "chap2_le0_ld10_H0_mixed":
            model = chap2_le0_ld10_H0_mixed
        case "chap2_le0_ld10_H500":
            model = chap2_le0_ld10_H500
        case "chap2_le10_ld5_H200":
            model = chap2_le10_ld5_H200
        case _:
            sys.exit(f"Model {args.model} doesn't exit!")

    # Seed randomness
    key = random.key(0)

    # Train or test
    if args.mode == "train":
        with open(args.config, "rb") as f:
            cfg = munch.munchify(tomllib.load(f))
        train(model, cfg, key)
    elif args.mode == "test":
        rundir = Path(args.rundir)
        test(model, rundir, key)


def train(model, cfg, key):
    # Get run directory path
    rundir = common.get_rundir(args.model, cfg)

    # Create run directory anew
    if rundir.exists():
        shutil.rmtree(rundir)
    rundir.mkdir(parents=True)

    # Dump config file in the run directory
    with open(rundir / "config.toml", "wb") as f:
        tomli_w.dump(cfg, f)

    # optimizer
    learning_rate = optax.linear_schedule(
        init_value=1e-2, end_value=1e-5, transition_steps=cfg.training.epochs
    )
    # learning_rate = optax.piecewise_constant_schedule(init_value=0.01,
    # boundaries_and_scales={10_000: 0.1, 20_000: 0.1})
    # learning_rate = 1e-3
    optim = optax.adam(learning_rate)

    # Module
    module = model.Module(cfg.module.hidden_width, cfg.module.num_hidden_layers)

    key, subkey = random.split(key)
    state = common.create_train_state(module, subkey, optim)

    @jax.jit
    def train_step(state, X):
        loss, grads = jax.value_and_grad(model.loss_fn)(state.params, state, X)
        state = state.apply_gradients(grads=grads)
        state = state.replace(loss=loss)
        return state

    @jax.jit
    def calc_valid_loss(state, X):
        return model.loss_fn(state.params, state, X)

    # Get train and validation data
    key, subkey = random.split(key)
    train_pts, valid_pts = common.get_datasets(subkey)

    epochs = cfg.training.epochs
    print_every = cfg.training.print_every
    loss_history = dict(train_loss=[], valid_loss=[])

    start_time = time.time()
    for epoch in range(epochs):
        state = train_step(state, train_pts)
        loss_history["train_loss"].append(state.loss.item())
        loss_history["valid_loss"].append(calc_valid_loss(state, valid_pts).item())

        if (epoch % print_every == 0) or (epoch == epochs - 1):
            print(
                f"{epoch=}, train_loss={loss_history["train_loss"][-1]}, valid_loss={loss_history["valid_loss"][-1]}"
            )
    print(f"Training over! Time taken {time.time() - start_time} s.")

#     opt = optax.chain(common.print_info(), optax.lbfgs(linesearch=optax.scale_by_zoom_linesearch(
#       max_linesearch_steps=100, verbose=True)))
#     init_state = state
#     print(
#     f'Initial value: {model.loss_fn(init_state.params, state, train_pts):.2e} '
#     f'Initial gradient norm: {otu.tree_l2_norm(jax.grad(model.loss_fn)(init_state.params, state, train_pts)):.2e}'
# )
#     final_state, _ = common.run_lbfgs(init_state, train_pts, model.loss_fn, opt, max_iter=50000, tol=1e-3)
#     print(
#         f'Final value: {model.loss_fn(final_state.params, final_state, train_pts):.2e}, '
#         f'Final gradient norm: {otu.tree_l2_norm(jax.grad(model.loss_fn)(final_state.params, final_state, train_pts)):.2e}'
#     )
#     state = final_state

    # Save the trained module
    with open(rundir / "params.pickle", "wb") as f:
        pickle.dump(serialization.to_state_dict(state.params), f)
    
    # Save loss history
    df_loss = pd.DataFrame.from_dict(loss_history)
    df_loss.to_csv(rundir / "loss_history.csv", index=False)

    # Plot loss history
    df_loss.plot(logy=True, title="Loss history")
    plt.savefig(rundir / "loss_history.pdf")


def test(model, rundir, key):
    with open(rundir / "config.toml", "rb") as f:
        cfg = munch.munchify(tomllib.load(f))

    # Module
    module = model.Module(cfg.module.hidden_width, cfg.module.num_hidden_layers)

    # Load the trained module
    with open(rundir / "params.pickle", "rb") as f:
        trained_dict = pickle.load(f)

    key, subkey = random.split(key)
    params = module.init(subkey, jnp.ones(2))["params"]
    params = serialization.from_state_dict(params, trained_dict)

    model.test(rundir, module, params)


if __name__ == "__main__":
    main()
