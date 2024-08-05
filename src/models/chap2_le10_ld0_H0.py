from pathlib import Path

import jax
import pandas as pd
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
from jax import random
from matplotlib import cm
from matplotlib import pyplot as plt

from src import common
from src.params import *

# Params
le = 10.0


class Module(nn.Module):
    hidden_width: int
    num_hidden_layers: int

    @nn.compact
    def __call__(self, X):
        widths = [self.hidden_width] * self.num_hidden_layers + [2]
        u, γp = common.MLP(widths)(X)

        t, x = X
        u = t * x * (1 - x) * u + t * x * umax / U
        γp *= t * x * (1 - x)

        return u, γp


def loss_fn(params, state, X):
    def loss_per_sample(x):
        _, γp = state.apply_fn({"params": params}, x)
        γp *= Γ
        du_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x)[0])
        du_dx = du_fn(x)[1] * U / L
        d2u_dx2 = jax.grad(lambda x: du_fn(x)[1])(x)[1] * U / L**2
        dγp_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x)[1])
        dγp = dγp_fn(x)
        γpdot, dγp_dx = dγp[0] * Γ / T, dγp[1] * Γ / L
        d2γp_dx2 = jax.grad(lambda x: dγp_fn(x)[1])(x)[1] * Γ / L**2

        dp = jnp.abs(γpdot)

        macro_loss = μ * (d2u_dx2 - dγp_dx)
        micro_loss = (
            μ * (du_dx - γp)
            - S0 * (dp / d0) ** m * jnp.sign(γpdot)
            + S0 * le**2 * d2γp_dx2
        )

        return macro_loss**2 + micro_loss**2

    return jnp.mean(jax.vmap(loss_per_sample)(X))


def test(rundir, module, params):
    ngrid = 100
    ts = jnp.linspace(t0, t1, ngrid)
    xs = jnp.linspace(x0, x1, ngrid)
    tgrid, xgrid = jnp.meshgrid(ts, xs)
    X = jnp.column_stack([tgrid.flatten(), xgrid.flatten()])

    u, γp = jax.vmap(lambda x: module.apply({"params": params}, x))(X)
    u *= U
    γp *= Γ

    du = jax.vmap(jax.grad(lambda x: module.apply({"params": params}, x)[0]))(X)
    du_dx = du[:, 1] * U / L
    τ = μ * (du_dx - γp)

    pd.DataFrame({"t": X[:, 0], "x": X[:, 1], "u": u, "γp": γp, "τ": τ}).to_csv(
        rundir / "grid_preds.csv", index=False
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xgrid * L, tgrid * T, u.reshape(xgrid.shape), cmap=cm.coolwarm
    )
    plt.title("u")
    plt.xlabel("x")
    plt.ylabel("t")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(rundir / "upred_over_grid.pdf")
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xgrid * L, tgrid * T, γp.reshape(xgrid.shape), cmap=cm.coolwarm
    )
    plt.title("γp")
    plt.xlabel("x")
    plt.ylabel("t")
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(rundir / "γppred_over_grid.pdf")
    plt.close()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(
        xgrid * L, tgrid * T, τ.reshape(xgrid.shape), cmap=cm.coolwarm
    )
    plt.title("τ")
    plt.xlabel("x")
    plt.ylabel("t")
    ax.set_zlim(0, 250)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(rundir / "τpred_over_grid.pdf")
    plt.close()

    X = jnp.column_stack([jnp.linspace(t0, t1, ngrid), jnp.ones(ngrid)])
    u, γp = jax.vmap(lambda x: module.apply({"params": params}, x))(X)
    u *= U
    γp *= Γ
    du = jax.vmap(jax.grad(lambda x: module.apply({"params": params}, x)[0]))(X)
    du_dx = du[:, 1] * U / L
    τ = μ * (du_dx - γp)
    γ = u / L
    df = pd.read_csv("fem_sols/anand_1d_fig_1a_curve3.csv")
    fig = plt.figure()
    plt.plot(df["x"], df["Curve3"], label="DAL")
    plt.plot(γ, τ, label="PIDL")
    plt.xlabel("Strain γ")
    plt.ylabel("Stress (MPa)")
    plt.legend()
    plt.savefig(rundir / "stress-strain.pdf")
    plt.close()

    x = jnp.linspace(x0, x1, ngrid)
    X = jnp.column_stack([jnp.ones(ngrid), x])
    _, γp = jax.vmap(lambda x: module.apply({"params": params}, x))(X)
    γp *= Γ
    df = pd.read_csv("fem_sols/anand_1d_fig_1b_curve3.csv")
    fig = plt.figure()
    plt.plot(df["Curve3"], df["x"], label="DAL")
    plt.plot(γp, x, label="PIDL")
    plt.xlim(left=0, right=0.15)
    plt.xlabel("Plastic strain γp")
    plt.ylabel("y/h")
    plt.legend()
    plt.savefig(rundir / "γp_profile.pdf")
    plt.close()
