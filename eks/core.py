from functools import partial

import jax
import jax.scipy as jsc
import numpy as np
import optax
from jax import jit
from jax import numpy as jnp
from jax import vmap
from typeguard import typechecked
from typing import List, Literal, Optional, Tuple, Union

from eks.marker_array import MarkerArray
from eks.utils import crop_frames

# -------------------------------------------------------------------------------------
# Kalman Functions: Functions related to performing filtering and smoothing
# -------------------------------------------------------------------------------------


@typechecked
def ensemble(
    marker_array: MarkerArray,
    avg_mode: Literal['mean', 'median'] = 'median',
    var_mode: Literal['var', 'confidence_weighted_var'] = 'confidence_weighted_var',
    nan_replacement: float = 1000.0
) -> MarkerArray:
    """
    Computes the ensemble mean (or median) and variance for a given MarkerArray.

    Aggregates predictions from multiple models to produce a single consensus MarkerArray
    with shape (1, n_cameras, n_frames, n_keypoints, 5),
    where the five fields are [x, y, var_x, var_y, likelihood].

    Args:
        marker_array: MarkerArray containing ensemble predictions.
            Shape (n_models, n_cameras, n_frames, n_keypoints, 3), with fields:
                ['x', 'y', 'likelihood'].
        avg_mode: Method to compute the central tendency of the ensemble.
            'median' | 'mean'
        var_mode: Method to compute ensemble variance.
            'var' — standard variance;
            'confidence_weighted_var' — variance scaled by inverse mean confidence.
        nan_replacement: Value used to replace NaNs in computed variance fields.

    Returns:
        MarkerArray of shape (1, n_cameras, n_frames, n_keypoints, 5), with fields:
            ['x', 'y', 'var_x', 'var_y', 'likelihood']
    """

    n_models, n_cameras, n_frames, n_keypoints, _ = marker_array.shape

    avg_func = jnp.nanmedian if avg_mode == 'median' else jnp.nanmean

    def compute_stats(data_x, data_y, data_lh):
        avg_x = avg_func(data_x, axis=0)
        avg_y = avg_func(data_y, axis=0)

        conf_per_keypoint = jnp.sum(data_lh, axis=0)
        mean_conf_per_keypoint = conf_per_keypoint / n_models

        var_x = jnp.nanvar(data_x, axis=0) / mean_conf_per_keypoint if var_mode in [
            'conf_weighted_var', 'confidence_weighted_var'] else jnp.nanvar(data_x, axis=0)
        var_y = jnp.nanvar(data_y, axis=0) / mean_conf_per_keypoint if var_mode in [
            'conf_weighted_var', 'confidence_weighted_var'] else jnp.nanvar(data_y, axis=0)

        # Replace NaNs in variance with chosen value
        var_x = jnp.nan_to_num(var_x, nan=nan_replacement)
        var_y = jnp.nan_to_num(var_y, nan=nan_replacement)

        return jnp.stack([avg_x, avg_y, var_x, var_y, mean_conf_per_keypoint], axis=-1)

    compute_stats_jit = jax.jit(compute_stats)

    # Unwrap MarkerArrays to JAX arrays and remove singleton field axis
    data_x = jnp.squeeze(jnp.array(marker_array.slice_fields("x").array), axis=-1)
    data_y = jnp.squeeze(jnp.array(marker_array.slice_fields("y").array), axis=-1)
    data_lh = jnp.squeeze(jnp.array(marker_array.slice_fields("likelihood").array), axis=-1)

    # Apply compute_stats in a single JIT call
    ensemble_array = np.array(compute_stats_jit(data_x, data_y, data_lh))
    ensemble_marker_array = MarkerArray(
        ensemble_array[None, ...],  # add n_models dim
        data_fields=['x', 'y', 'var_x', 'var_y', 'likelihood']
    )

    return ensemble_marker_array

@typechecked
def kalman_filter_step(
        carry,
        inputs
) -> Tuple[tuple, Tuple[jnp.ndarray, jnp.ndarray, jax.Array]]:
    """
    Performs a single Kalman filter update step using time-varying observation noise
    from ensemble variance.

    Used in a scan loop, updating the state mean and covariance
    based on the current observation and its associated ensemble variance.

    Args:
        carry: Tuple containing the previous state and model parameters:
            - m_prev (jnp.ndarray): Previous state estimate (mean vector).
            - V_prev (jnp.ndarray): Previous state covariance matrix.
            - A (jnp.ndarray): State transition matrix.
            - Q (jnp.ndarray): Process noise covariance matrix.
            - C (jnp.ndarray): Observation matrix.
            - nll_net (float): Accumulated negative log-likelihood.
        inputs: Tuple containing the current observation and its estimated ensemble variance:
            - curr_y (jnp.ndarray): Current observation vector.
            - curr_ensemble_var (jnp.ndarray): Estimated observation noise variance
                                               (used to build time-varying R matrix).

    Returns:
        A tuple of two elements:
            - carry (tuple): Updated (m_t, V_t, A, Q, C, nll_net) to pass to the next step.
            - output (tuple): Tuple of:
                - m_t (jnp.ndarray): Updated state mean.
                - V_t (jnp.ndarray): Updated state covariance.
                - nll_current (float, stored as jax.Array): NLL of the current observation.
    """
    m_prev, V_prev, A, Q, C, nll_net = carry
    curr_y, curr_ensemble_var = inputs

    # Update R with time-varying ensemble variance
    R = jnp.diag(curr_ensemble_var)

    # Predict
    m_pred = jnp.dot(A, m_prev)
    V_pred = jnp.dot(A, jnp.dot(V_prev, A.T)) + Q

    # Update
    innovation = curr_y - jnp.dot(C, m_pred)
    innovation_cov = jnp.dot(C, jnp.dot(V_pred, C.T)) + R
    K = jnp.dot(V_pred, jnp.dot(C.T, jnp.linalg.inv(innovation_cov)))
    m_t = m_pred + jnp.dot(K, innovation)
    V_t = jnp.dot((jnp.eye(V_pred.shape[0]) - jnp.dot(K, C)), V_pred)

    nll_current = single_timestep_nll(innovation, innovation_cov)
    nll_net = nll_net + nll_current

    return (m_t, V_t, A, Q, C, nll_net), (m_t, V_t, nll_current)


@typechecked
def kalman_filter_step_nlls(
    carry: tuple,
    inputs: tuple
) -> Tuple[tuple, Tuple[jnp.ndarray, jnp.ndarray, float]]:
    """
    Performs a single Kalman filter update step and records per-timestep negative
    log-likelihoods (NLLs) into a preallocated array.

    Used inside a `lax.scan` loop. In addition to updating the state estimate and total NLL,
    it writes the NLL of each timestep into a persistent array for later analysis/plotting.

    Args:
        carry: Tuple containing:
            - m_prev (jnp.ndarray): Previous state estimate (mean vector).
            - V_prev (jnp.ndarray): Previous state covariance matrix.
            - A (jnp.ndarray): State transition matrix.
            - Q (jnp.ndarray): Process noise covariance matrix.
            - C (jnp.ndarray): Observation matrix.
            - nll_net (float): Cumulative negative log-likelihood.
            - nll_array (jnp.ndarray): Preallocated array for per-step NLL values.
            - t (int): Current timestep index into the NLL array.

        inputs: Tuple containing:
            - curr_y (jnp.ndarray): Current observation vector.
            - curr_ensemble_var (jnp.ndarray): Estimated observation noise variance,
                                               used to construct the time-varying R matrix.

    Returns:
        A tuple of:
            - carry (tuple): Updated state and NLL tracking info for the next timestep.
            - output (tuple):
                - m_t (jnp.ndarray): Updated state mean.
                - V_t (jnp.ndarray): Updated state covariance.
                - nll_current (float): Negative log-likelihood of the current timestep.
    """
    # Unpack carry and inputs
    m_prev, V_prev, A, Q, C, nll_net, nll_array, t = carry
    curr_y, curr_ensemble_var = inputs

    # Update R with the current ensemble variance
    R = jnp.diag(curr_ensemble_var)

    # Predict
    m_pred = jnp.dot(A, m_prev)
    V_pred = jnp.dot(A, jnp.dot(V_prev, A.T)) + Q

    # Update
    innovation = curr_y - jnp.dot(C, m_pred)
    innovation_cov = jnp.dot(C, jnp.dot(V_pred, C.T)) + R
    K = jnp.dot(V_pred, jnp.dot(C.T, jnp.linalg.inv(innovation_cov)))
    m_t = m_pred + jnp.dot(K, innovation)
    V_t = V_pred - jnp.dot(K, jnp.dot(C, V_pred))

    # Compute the negative log-likelihood for the current time step
    nll_current = single_timestep_nll(innovation, innovation_cov)

    # Accumulate the negative log-likelihood
    nll_net = nll_net + nll_current

    # Save the current NLL to the preallocated array
    nll_array = nll_array.at[t].set(nll_current)

    # Increment the time step
    t = t + 1

    # Return the updated state and outputs
    return (m_t, V_t, A, Q, C, nll_net, nll_array, t), (m_t, V_t, nll_current)


@partial(jit, backend='cpu')
def forward_pass(
    y: jnp.ndarray,
    m0: jnp.ndarray,
    cov0: jnp.ndarray,
    A: jnp.ndarray,
    Q: jnp.ndarray,
    C: jnp.ndarray,
    ensemble_vars: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Executes the Kalman filter forward pass for a single keypoint over time,
    incorporating time-varying observation noise variances.

    Computes filtered state means, covariances, and the cumulative
    negative log-likelihood across all timesteps. Used within `vmap` to
    handle multiple keypoints in parallel.

    Args:
        y: Array of shape (T, obs_dim). Sequence of observations over time.
        m0: Array of shape (state_dim,). Initial state estimate.
        cov0: Array of shape (state_dim, state_dim). Initial state covariance.
        A: Array of shape (state_dim, state_dim). State transition matrix.
        Q: Array of shape (state_dim, state_dim). Process noise covariance matrix.
        C: Array of shape (obs_dim, state_dim). Observation matrix.
        ensemble_vars: Array of shape (T, obs_dim). Per-frame observation noise variances.

    Returns:
        mfs: Array of shape (T, state_dim). Filtered mean estimates at each timestep.
        Vfs: Array of shape (T, state_dim, state_dim). Filtered covariance estimates at each timestep.
        nll_net: Scalar float. Total negative log-likelihood across all timesteps.
    """
    # Initialize carry
    carry = (m0, cov0, A, Q, C, 0)
    # Run the scan, passing y and ensemble_vars as inputs to kalman_filter_step
    carry, outputs = jax.lax.scan(kalman_filter_step, carry, (y, ensemble_vars))
    mfs, Vfs, _ = outputs
    nll_net = carry[-1]
    return mfs, Vfs, nll_net


@typechecked
def kalman_smoother_step(
    carry: tuple,
    X: list,
) -> Tuple[tuple, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Performs a single backward pass of the Kalman smoother.

    Updates the smoothed state estimate and covariance based on the
    current filtered estimate and the next time step's smoothed estimate. Used
    within a `jax.lax.scan` in reverse over the time axis.

    Args:
        carry: Tuple containing:
            - m_ahead_smooth (jnp.ndarray): Smoothed state mean at the next timestep.
            - v_ahead_smooth (jnp.ndarray): Smoothed state covariance at the next timestep.
            - A (jnp.ndarray): State transition matrix.
            - Q (jnp.ndarray): Process noise covariance matrix.

        X: Tuple containing:
            - m_curr_filter (jnp.ndarray): Filtered mean estimate at the current timestep.
            - v_curr_filter (jnp.ndarray): Filtered covariance at the current timestep.

    Returns:
        A tuple of:
            - carry (tuple): Updated smoothed state (mean, cov) and model params for the next step.
            - output (tuple):
                - smoothed_state (jnp.ndarray): Smoothed mean estimate at the current timestep.
                - smoothed_cov (jnp.ndarray): Smoothed covariance at the current timestep.
    """
    m_ahead_smooth, v_ahead_smooth, A, Q = carry
    m_curr_filter, v_curr_filter = X[0], X[1]

    # Compute the smoother gain
    ahead_cov = jnp.dot(A, jnp.dot(v_curr_filter, A.T)) + Q

    smoothing_gain = jsc.linalg.solve(ahead_cov, jnp.dot(A, v_curr_filter.T)).T
    smoothed_state = m_curr_filter + jnp.dot(smoothing_gain, m_ahead_smooth - m_curr_filter)
    smoothed_cov = v_curr_filter + jnp.dot(jnp.dot(smoothing_gain, v_ahead_smooth - ahead_cov),
                                           smoothing_gain.T)

    return (smoothed_state, smoothed_cov, A, Q), (smoothed_state, smoothed_cov)


# @typechecked -- raises InstrumentationWarning as @jit rewrites into compiled form (JAX XLA)
@partial(jit, backend='cpu')
def backward_pass(
    mfs: jnp.ndarray,
    Vfs: jnp.ndarray,
    A: jnp.ndarray,
    Q: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Executes the Kalman smoother backward pass using filtered means and covariances.

    Refines forward-filtered estimates by incorporating future observations.
    Used after a Kalman filter forward pass to recover more accurate state estimates.

    Args:
        mfs: Array of shape (T, state_dim). Filtered state means from the forward pass.
        Vfs: Array of shape (T, state_dim, state_dim). Filtered covariances from the forward pass.
        A: Array of shape (state_dim, state_dim). State transition matrix.
        Q: Array of shape (state_dim, state_dim). Process noise covariance matrix.

    Returns:
        smoothed_states: Array of shape (T, state_dim). Smoothed state mean estimates.
        smoothed_state_covariances: Array of shape (T, state_dim, state_dim).
            Smoothed state covariance estimates.
    """
    carry = (mfs[-1], Vfs[-1], A, Q)

    # Reverse scan over the time steps
    carry, outputs = jax.lax.scan(
        kalman_smoother_step,
        carry,
        [mfs[:-1], Vfs[:-1]],
        reverse=True
    )

    smoothed_states, smoothed_state_covariances = outputs
    smoothed_states = jnp.append(smoothed_states, jnp.expand_dims(mfs[-1], 0), 0)
    smoothed_state_covariances = jnp.append(smoothed_state_covariances,
                                            jnp.expand_dims(Vfs[-1], 0), 0)
    return smoothed_states, smoothed_state_covariances


@typechecked
def single_timestep_nll(
    innovation: jnp.ndarray,
    innovation_cov: jnp.ndarray
) -> jax.Array:
    """
    Computes the negative log-likelihood (NLL) of a single multivariate Gaussian observation.

    Measures how well the predicted state explains the current observation.
    A small regularization term (epsilon) is added to the covariance to ensure numerical stability.

    Args:
        innovation: Array of shape (D,). The difference between observed and predicted observation.
        innovation_cov: Array of shape (D, D). Covariance of the innovation.

    Returns:
        nll_increment: Scalar float stored as a jax.Array.
                       Negative log-likelihood of observing the current innovation.
    """
    epsilon = 1e-6
    n_coords = innovation.shape[0]

    # Regularize the innovation covariance matrix by adding epsilon to the diagonal
    reg_innovation_cov = innovation_cov + epsilon * jnp.eye(n_coords)

    # Compute the log determinant of the regularized covariance matrix
    log_det_S = jnp.log(jnp.abs(jnp.linalg.det(reg_innovation_cov)) + epsilon)
    solved_term = jnp.linalg.solve(reg_innovation_cov, innovation)
    quadratic_term = jnp.dot(innovation, solved_term)

    # Compute the NLL increment for the current time step
    c = jnp.log(2 * jnp.pi) * n_coords  # The Gaussian normalization constant part
    nll_increment = 0.5 * jnp.abs(log_det_S + quadratic_term + c)
    return nll_increment


@typechecked
def final_forwards_backwards_pass(
    process_cov: jnp.ndarray,
    s: np.ndarray,
    ys: np.ndarray,
    m0s: jnp.ndarray,
    S0s: jnp.ndarray,
    Cs: jnp.ndarray,
    As: jnp.ndarray,
    ensemble_vars: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the full Kalman forward-backward smoother across all keypoints using
    optimized smoothing parameters.

    Computes smoothed state means and covariances for each keypoint over time.
    The process noise covariance is scaled per-keypoint by a learned smoothing parameter `s`.

    Args:
        process_cov: Array of shape (K, D, D). Base process noise covariance per keypoint.
        s: Array of shape (K,). Smoothing scalars applied to process_cov per keypoint.
        ys: Array of shape (K, T, obs_dim). Observations per keypoint over time.
        m0s: Array of shape (K, D). Initial state mean per keypoint.
        S0s: Array of shape (K, D, D). Initial state covariance per keypoint.
        Cs: Array of shape (K, obs_dim, D). Observation matrix per keypoint.
        As: Array of shape (K, D, D). State transition matrix per keypoint.
        ensemble_vars: Array of shape (T, K, obs_dim). Time-varying obs variances per keypoint.

    Returns:
        smoothed_means: Array of shape (K, T, D). Smoothed state means for each keypoint over time.
        smoothed_covariances: Array of shape (K, T, D, D). Smoothed state covariances over time.
    """

    # Initialize
    n_keypoints = ys.shape[0]
    ms_array = []
    Vs_array = []
    Qs = s[:, None, None] * process_cov

    # Run forward and backward pass for each keypoint
    for k in range(n_keypoints):
        mf, Vf, nll = forward_pass(
            ys[k], m0s[k], S0s[k], As[k], Qs[k], Cs[k], ensemble_vars[:, k, :])
        ms, Vs = backward_pass(mf, Vf, As[k], Qs[k])

        ms_array.append(np.array(ms))
        Vs_array.append(np.array(Vs))

    smoothed_means = np.stack(ms_array, axis=0)
    smoothed_covariances = np.stack(Vs_array, axis=0)

    return smoothed_means, smoothed_covariances

# -------------------------------------------------------------------------------------
# Optimization: Functions related to optimizing the smoothing hyperparameter
# -------------------------------------------------------------------------------------


@typechecked
def compute_initial_guesses(
    ensemble_vars: Union[np.ndarray, list]
) -> float:
    """
    Computes an initial guess for the smoothing parameter `s` by estimating
    the temporal variability of the ensemble variance.

    Computes the standard deviation of frame-to-frame changes
    in ensemble variance, clipped to the first 2000 frames for stability.

    Args:
        ensemble_vars: Array of shape (T, K, D), where:
            - T is the number of frames (timepoints),
            - K is the number of keypoints,
            - D is the number of observation dimensions (usually 2).

    Returns:
        std_dev_guess: A float representing the initial guess for the smoothing parameter,
                       based on temporal standard deviation of ensemble variance.
    """
    ensemble_vars = np.asarray(ensemble_vars)[:2000]

    if ensemble_vars.shape[0] < 2:
        raise ValueError("Not enough frames to compute temporal differences.")

    # Compute temporal differences
    temporal_diffs = ensemble_vars[1:] - ensemble_vars[:-1]

    # Compute standard deviation across all temporal differences
    std_dev_guess = round(np.nanstd(temporal_diffs), 5)
    return float(std_dev_guess)


@typechecked
def optimize_smooth_param(
    cov_mats: jnp.ndarray,
    ys: np.ndarray,
    m0s: jnp.ndarray,
    S0s: jnp.ndarray,
    Cs: jnp.ndarray,
    As: jnp.ndarray,
    ensemble_vars: np.ndarray,
    s_frames: Optional[List] = None,
    smooth_param: Optional[Union[float, List[float]]] = None,
    blocks: Optional[List[List[int]]] = None,
    maxiter: int = 1000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize smoothing parameters for each keypoint (or block of keypoints) using
    negative log-likelihood minimization, and apply final Kalman forward-backward smoothing.

    If `smooth_param` is provided, it is used directly. Otherwise, the function computes
    initial guesses and uses gradient descent to optimize per-block values of `s`.

    Args:
        cov_mats: Array of shape (K, D, D). Base process noise covariances per keypoint.
        ys: Array of shape (K, T, obs_dim). Observations per keypoint over time.
        m0s: Array of shape (K, D). Initial state means per keypoint.
        S0s: Array of shape (K, D, D). Initial state covariances per keypoint.
        Cs: Array of shape (K, obs_dim, D). Observation matrices per keypoint.
        As: Array of shape (K, D, D). State transition matrices per keypoint.
        ensemble_vars: Array of shape (T, K, obs_dim). Time-varying ensemble variances.
        s_frames: Optional list of frame indices for computing initial guess statistics.
        smooth_param: Optional fixed value(s) of smoothing param `s`.
                      Can be a float or list of floats (one per keypoint/block).
        blocks: Optional list of lists of keypoint indices to share a smoothing param.
                Defaults to treating each keypoint independently.
        maxiter: Max number of optimization steps per block.
        verbose: If True, print progress logs.

    Returns:
        s_finals: Array of shape (K,). Final smoothing parameter per keypoint.
        ms: Array of shape (K, T, D). Smoothed state means.
        Vs: Array of shape (K, T, D, D). Smoothed state covariances.
    """

    n_keypoints = ys.shape[0]
    s_finals = []
    if blocks is None:
        blocks = []
    if len(blocks) == 0:
        for n in range(n_keypoints):
            blocks.append([n])
    if verbose:
        print(f'Correlated keypoint blocks: {blocks}')

    @partial(jit)
    def nll_loss_sequential_scan(s, cov_mats, cropped_ys, m0s, S0s, Cs, As, ensemble_vars):
        s = jnp.exp(s)  # To ensure positivity
        return smooth_min(
            s, cov_mats, cropped_ys, m0s, S0s, Cs, As, ensemble_vars)

    loss_function = nll_loss_sequential_scan

    # Optimize smooth_param
    if smooth_param is not None:
        if isinstance(smooth_param, float):
            s_finals = [smooth_param]
        elif isinstance(smooth_param, int):
            s_finals = [float(smooth_param)]
        else:
            s_finals = smooth_param
    else:
        guesses = []
        cropped_ys = []
        for k in range(n_keypoints):
            current_guess = compute_initial_guesses(ensemble_vars[:, k, :])
            guesses.append(current_guess)
            if s_frames is None or len(s_frames) == 0:
                cropped_ys.append(ys[k])
            else:
                cropped_ys.append(crop_frames(ys[k], s_frames))

        cropped_ys = np.array(cropped_ys)  # Concatenation of this list along dimension 0

        # Optimize negative log likelihood
        for block in blocks:
            s_init = guesses[block[0]]
            if s_init <= 0:
                s_init = 2
            s_init = jnp.log(s_init)
            optimizer = optax.adam(learning_rate=0.25)
            opt_state = optimizer.init(s_init)

            selector = np.array(block).astype(int)
            cov_mats_sub = cov_mats[selector]
            m0s_crop = m0s[selector]
            S0s_crop = S0s[selector]
            Cs_crop = Cs[selector]
            As_crop = As[selector]
            y_subset = cropped_ys[selector]
            ensemble_vars_crop = np.swapaxes(ensemble_vars[:, selector, :], 0, 1)

            def step(s, opt_state):
                loss, grads = jax.value_and_grad(loss_function)(
                    s, cov_mats_sub, y_subset, m0s_crop, S0s_crop, Cs_crop, As_crop,
                    ensemble_vars_crop)
                updates, opt_state = optimizer.update(grads, opt_state)
                s = optax.apply_updates(s, updates)
                return s, opt_state, loss

            prev_loss = jnp.inf
            for iteration in range(maxiter):
                s_init, opt_state, loss = step(s_init, opt_state)

                if verbose and iteration % 10 == 0 or iteration == maxiter - 1:
                    print(f'Iteration {iteration}, Current loss: {loss}, Current s: {s_init}')

                tol = 0.001 * jnp.abs(jnp.log(prev_loss))
                if jnp.linalg.norm(loss - prev_loss) < tol + 1e-6:
                    break
                prev_loss = loss

            s_final = jnp.exp(s_init)  # Convert back from log-space

            for b in block:
                if verbose:
                    print(f's={s_final} for keypoint {b}')
                s_finals.append(s_final)

    s_finals = np.array(s_finals)
    # Final smooth with optimized s
    ms, Vs = final_forwards_backwards_pass(
        cov_mats, s_finals, ys, m0s, S0s, Cs, As, ensemble_vars,
    )

    return s_finals, ms, Vs


@typechecked
def inner_smooth_min_routine(
    y: jnp.ndarray,
    m0: jnp.ndarray,
    S0: jnp.ndarray,
    A: jnp.ndarray,
    Q: jnp.ndarray,
    C: jnp.ndarray,
    ensemble_var: jnp.ndarray
) -> jax.Array:
    # Run filtering with the current smooth_param
    _, _, nll = forward_pass(y, m0, S0, A, Q, C, ensemble_var)
    return nll


inner_smooth_min_routine_vmap = vmap(inner_smooth_min_routine, in_axes=(0, 0, 0, 0, 0, 0, 0))


@typechecked
def smooth_min(
    smooth_param: jax.Array,
    cov_mats: jnp.ndarray,
    ys: jnp.ndarray,
    m0s: jnp.ndarray,
    S0s: jnp.ndarray,
    Cs: jnp.ndarray,
    As: jnp.ndarray,
    ensemble_vars: jnp.ndarray
) -> jax.Array:
    """
    Computes the total negative log-likelihood (NLL) for a given smoothing parameter
    by running a full forward-pass Kalman filter over all keypoints.

    This is the objective function minimized during smoothing parameter optimization.

    Args:
        smooth_param: Scalar float value of the smoothing parameter `s`.
        cov_mats: Array of shape (K, D, D). Process noise covariance templates.
        ys: Array of shape (K, T, obs_dim). Observations per keypoint.
        m0s: Array of shape (K, D). Initial state means.
        S0s: Array of shape (K, D, D). Initial state covariances.
        Cs: Array of shape (K, obs_dim, D). Observation matrices.
        As: Array of shape (K, D, D). State transition matrices.
        ensemble_vars: Array of shape (T, K, obs_dim). Time-varying ensemble variances.

    Returns:
        nlls: Scalar JAX array. Total negative log-likelihood across all keypoints.
    """
    # Adjust Q based on smooth_param and cov_matrix
    Qs = smooth_param * cov_mats
    nlls = jnp.sum(inner_smooth_min_routine_vmap(ys, m0s, S0s, As, Qs, Cs, ensemble_vars))
    return nlls
