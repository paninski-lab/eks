from dynamax.nonlinear_gaussian_ssm.inference_ekf import extended_kalman_smoother
from dynamax.nonlinear_gaussian_ssm.models import (
    ParamsNLGSSM,
)

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Tuple, Callable
from typeguard import typechecked

ArrayLike = Union[np.ndarray, jax.Array]

@typechecked
def dynamax_ekf_smooth_routine(
    y: ArrayLike,
    m0: ArrayLike,
    S0: ArrayLike,
    A: ArrayLike,
    Q: ArrayLike,
    C: ArrayLike | None,
    ensemble_vars: ArrayLike,  # shape (T, obs_dim)
    f_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    h_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extended Kalman smoother using the Dynamax nonlinear interface,
    allowing for time-varying observation noise.

    By default, uses linear dynamics and emissions: f(x) = Ax, h(x) = Cx.

    Args:
        y: (T, obs_dim) observation sequence.
        m0: (state_dim,) initial mean.
        S0: (state_dim, state_dim) initial covariance.
        A: (state_dim, state_dim) dynamics matrix.
        Q: (state_dim, state_dim) process noise covariance.
        C: (obs_dim, state_dim) emission matrix (optional).
        ensemble_vars: (T, obs_dim) per-timestep observation noise variance.
        f_fn: optional dynamics function f(x).
        h_fn: optional emission function h(x).

    Returns:
        smoothed_means: (T, state_dim)
        smoothed_covariances: (T, state_dim, state_dim)
    """
    y, m0, S0, A, Q, ensemble_vars = map(jnp.asarray, (y, m0, S0, A, Q, ensemble_vars))
    C = jnp.asarray(C) if C is not None else None

    if f_fn is None:
        f_fn = lambda x: A @ x
    if h_fn is None:
        if C is None:
            raise ValueError("Must provide either emission matrix C or a nonlinear emission function h_fn.")
        h_fn = lambda x: C @ x

    # Dynamically determine obs_dim from h_fn output
    obs_dim = y.shape[1]
    R_t = jnp.stack([jnp.diag(var_t[:obs_dim]) for var_t in ensemble_vars], axis=0)  # shape (T, obs_dim, obs_dim)
    params = ParamsNLGSSM(
        initial_mean=m0,
        initial_covariance=S0,
        dynamics_function=f_fn,
        dynamics_covariance=Q,
        emission_function=h_fn,
        emission_covariance=R_t,
    )

    posterior = extended_kalman_smoother(params, y)
    return posterior.smoothed_means, posterior.smoothed_covariances