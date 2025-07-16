from dynamax.linear_gaussian_ssm.models import (
    LinearGaussianSSM,
    ParamsLGSSM,
    ParamsLGSSMInitial,
    ParamsLGSSMDynamics,
    ParamsLGSSMEmissions
)
import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Tuple
from typeguard import typechecked

ArrayLike = Union[np.ndarray, jax.Array]

@typechecked
def dynamax_linear_smooth_routine(
    y: ArrayLike,
    m0: ArrayLike,
    S0: ArrayLike,
    A: ArrayLike,
    Q: ArrayLike,
    C: ArrayLike,
    ensemble_vars: ArrayLike  # shape (T, obs_dim)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Run Dynamax smoother with time-varying diagonal observation noise from ensemble variances.

    Args:
        y: (T, obs_dim) observations
        m0: (state_dim,) initial mean
        S0: (state_dim, state_dim) initial covariance
        A: (state_dim, state_dim) transition matrix
        Q: (state_dim, state_dim) process noise
        C: (obs_dim, state_dim) emission matrix
        ensemble_vars: (T, obs_dim) per-timestep observation noise variance

    Returns:
        smoothed_means: (T, state_dim)
        smoothed_covs: (T, state_dim, state_dim)
    """

    # Convert everything to JAX arrays
    y, m0, S0, A, Q, C, ensemble_vars = map(jnp.asarray, (y, m0, S0, A, Q, C, ensemble_vars))
    T, obs_dim = y.shape
    state_dim = A.shape[0]

    # Dynamax accepts time-varying diagonal R_t as (T, obs_dim)
    model = LinearGaussianSSM(state_dim, obs_dim)

    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(mean=m0, cov=S0),
        dynamics=ParamsLGSSMDynamics(
            weights=A,
            cov=Q,
            bias=jnp.zeros(state_dim),
            input_weights=jnp.zeros((state_dim, 0))
        ),
        emissions=ParamsLGSSMEmissions(
            weights=C,
            cov=ensemble_vars,  # <=== time-varying diagonal noise
            bias=jnp.zeros(obs_dim),
            input_weights=jnp.zeros((obs_dim, 0))
        )
    )

    posterior = model.smoother(params, y)
    return posterior.smoothed_means, posterior.smoothed_covariances
