import jax
import pandas as pd
from flax import linen as nn
from jax import numpy as jnp
from matplotlib import cm
from matplotlib import pyplot as plt

from src import common
from src.params import *

## Params
ld = 10
Σ = S0


class Module(nn.Module):
    hidden_width: int
    num_hidden_layers: int

    @nn.compact
    def __call__(self, X):
        widths = [self.hidden_width] * self.num_hidden_layers + [3]
        u, γp, kp = common.MLP(widths)(X)

        t, x = X
        u = t * x * (1 - x) * u + t * x * umax / U
        γp *= t * x * (1 - x)
        kp *= t

        return u, γp, kp


def loss_fn(params, state, X):
    def loss_per_sample(x):
        _, γp, Kp = state.apply_fn({"params": params}, x)
        γp *= Γ
        Kp *= Σ
        dKp_dx = (
            jax.grad(lambda x: state.apply_fn({"params": params}, x)[2])(x)[1] * Σ / L
        )
        du_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x)[0])
        du_dx = du_fn(x)[1] * U / L
        d2u_dx2 = jax.grad(lambda x: du_fn(x)[1])(x)[1] * U / L**2
        dγp_fn = jax.grad(lambda x: state.apply_fn({"params": params}, x)[1])
        dγp = dγp_fn(x)
        γpdot, dγp_dx = dγp[0] * Γ / T, dγp[1] * Γ / L
        dγpdot_dx = jax.grad(lambda x: dγp_fn(x)[0])(x)[1] * Γ / T / L

        dp = jnp.sqrt(γpdot**2 + ld**2 * dγpdot_dx**2)

        macro_loss = μ * (d2u_dx2 - dγp_dx)
        micro_loss = μ * (du_dx - γp) - S0 * (dp / d0) ** m * jnp.sign(γpdot) + dKp_dx
        microstress_loss = Kp - S0 * ld**2 * (dp / d0) ** m * dγpdot_dx / dp

        return (macro_loss) ** 2 + (micro_loss) ** 2 + (microstress_loss) ** 2

    return jnp.mean(jax.vmap(loss_per_sample)(X))


def test(rundir, module, params):
    ngrid = 100
    ts = jnp.linspace(t0, t1, ngrid)
    xs = jnp.linspace(x0, x1, ngrid)
    tgrid, xgrid = jnp.meshgrid(ts, xs)
    X = jnp.column_stack([tgrid.flatten(), xgrid.flatten()])

    u, γp, _ = jax.vmap(lambda x: module.apply({"params": params}, x))(X)
    u = u.flatten() * U
    γp = γp.flatten() * Γ

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
    ax.set_zlim(0, 400)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.savefig(rundir / "τpred_over_grid.pdf")
    plt.close()

    X = jnp.column_stack([jnp.linspace(t0, t1, ngrid), jnp.ones(ngrid)])
    u, γp, _ = jax.vmap(lambda x: module.apply({"params": params}, x))(X)
    u = u.flatten() * U
    γp = γp.flatten() * Γ
    du = jax.vmap(jax.grad(lambda x: module.apply({"params": params}, x)[0]))(X)
    du_dx = du[:, 1] * U / L
    τ = μ * (du_dx - γp)
    γ = u / L
    df = pd.read_csv("fem_sols/anand_1d_fig_3a_curve3.csv")
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
    _, γp, _ = jax.vmap(lambda x: module.apply({"params": params}, x))(X)
    γp *= Γ
    df = pd.read_csv("fem_sols/anand_1d_fig_3b_curve3.csv")
    fig = plt.figure()
    plt.plot(df["Curve3"], df["x"], label="DAL")
    plt.plot(γp, x, label="PIDL")
    plt.xlim(left=0, right=0.15)
    plt.xlabel("Plastic strain γp")
    plt.ylabel("y/h")
    plt.legend()
    plt.savefig(rundir / "γp_profile.pdf")
    plt.close()
