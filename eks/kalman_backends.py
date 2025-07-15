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
    R: ArrayLike
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    # Convert everything to JAX arrays
    y, m0, S0, A, Q, C, R = map(jnp.asarray, (y, m0, S0, A, Q, C, R))
    state_dim, obs_dim = A.shape[0], C.shape[0]

    # Build model and correct param structure
    model = LinearGaussianSSM(state_dim, obs_dim)

    params = ParamsLGSSM(
        initial=ParamsLGSSMInitial(mean=m0, cov=S0),
        dynamics=ParamsLGSSMDynamics(
            weights=A,
            cov=Q,
            bias=jnp.zeros(A.shape[0]),  # shape (state_dim,)
            input_weights=jnp.zeros((A.shape[0], 0))  # shape (state_dim, 0) for no control input
        ),
        emissions=ParamsLGSSMEmissions(
            weights=C,
            cov=R,
            bias=jnp.zeros(C.shape[0]),  # shape (obs_dim,)
            input_weights=jnp.zeros((C.shape[0], 0))  # shape (obs_dim, 0) for no control input
        )
    )

    posterior = model.smoother(params, y)
    return posterior.smoothed_means, posterior.smoothed_covariances
