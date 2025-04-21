from functools import partial

import jax
import jax.scipy as jsc
import numpy as np
import optax
from jax import jit, vmap
from jax import numpy as jnp
from typeguard import typechecked

from eks.marker_array import MarkerArray

# ------------------------------------------------------------------------------------------
# Original Core Functions: These functions are still in use for the IBL scripts
# as of this update, but will eventually be replaced the with JAX versions
# ------------------------------------------------------------------------------------------
from eks.utils import crop_frames


@typechecked
def ensemble(
    markers_list: list,
    keys: list,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
) -> tuple:
    """Compute ensemble mean/median and variance of marker dataframes.

    Args:
        markers_list: List of DLC marker dataframes
        keys: List of keys in each marker dataframe
        avg_mode
            'median' | 'mean'
        var_mode
            'confidence_weighted_var' | 'var'

    Returns:
        tuple:
            ensemble_preds: np.ndarray
                shape (samples, n_keypoints)
            ensemble_vars: np.ndarray
                shape (samples, n_keypoints)
            ensemble_likelihoods: np.ndarray
                shape (samples, n_keypoints)
            ensemble_stacks: np.ndarray
                shape (n_models, samples, n_keypoints)

    """

    ensemble_preds = []
    ensemble_vars = []
    ensemble_likes = []
    ensemble_stacks = []

    if avg_mode == 'median':
        average_func = np.nanmedian
    elif avg_mode == 'mean':
        average_func = np.nanmean
    else:
        raise ValueError(f"avg_mode={avg_mode} not supported")

    for key in keys:

        # compute mean/median
        stack = np.zeros((markers_list[0].shape[0], len(markers_list)))
        for k in range(len(markers_list)):
            stack[:, k] = markers_list[k][key]
        ensemble_stacks.append(stack)
        avg = average_func(stack, axis=1)
        ensemble_preds.append(avg)

        # collect likelihoods
        likelihood_stack = np.ones((markers_list[0].shape[0], len(markers_list)))
        likelihood_key = key[:-1] + 'likelihood'
        if likelihood_key in markers_list[0]:
            for k in range(len(markers_list)):
                likelihood_stack[:, k] = markers_list[k][likelihood_key]
        mean_conf_per_keypoint = np.mean(likelihood_stack, axis=1)
        ensemble_likes.append(mean_conf_per_keypoint)

        # compute variance
        var = np.nanvar(stack, axis=1)
        if var_mode in ['conf_weighted_var', 'confidence_weighted_var']:
            var = var / mean_conf_per_keypoint  # low-confidence --> inflated obs variances
        elif var_mode != 'var':
            raise ValueError(f"var_mode={var_mode} not supported")
        ensemble_vars.append(var)

    ensemble_preds = np.asarray(ensemble_preds).T
    ensemble_vars = np.asarray(ensemble_vars).T
    ensemble_likes = np.asarray(ensemble_likes).T
    ensemble_stacks = np.asarray(ensemble_stacks).T

    return ensemble_preds, ensemble_vars, ensemble_likes, ensemble_stacks


def forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars):
    """Implements Kalman-filter
    Args:
        y: np.ndarray
            shape (samples, n_keypoints)
        m0: np.ndarray
            shape (n_latents)
        S0: np.ndarray
            shape (n_latents, n_latents)
        C: np.ndarray
            shape (n_keypoints, n_latents)
        R: np.ndarray
            shape (n_keypoints, n_keypoints)
        A: np.ndarray
            shape (n_latents, n_latents)
        Q: np.ndarray
            shape (n_latents, n_latents)
        ensemble_vars: np.ndarray
            shape (samples, n_keypoints)

    Returns:
        mfs: np.ndarray
            shape (samples, n_keypoints)
        Vfs: np.ndarray
            shape (samples, n_latents, n_latents)
        Ss: np.ndarray
            shape (samples, n_latents, n_latents)
        innovations: np.ndarray
            shape (samples, n_keypoints)
        innovation_cov: np.ndarray
            shape (samples, n_keypoints, n_keypoints)
    """
    T = y.shape[0]
    mf = np.zeros(shape=(T, m0.shape[0]))
    Vf = np.array([np.eye(m0.shape[0]) for _ in range(T)])
    S = np.array([np.eye(m0.shape[0]) for _ in range(T)])
    innovations = np.zeros((T, y.shape[1]))
    innovation_cov = np.zeros((T, C.shape[0], C.shape[0]))

    # time-varying observation variance
    for i in range(ensemble_vars.shape[1]):
        R[i, i] = ensemble_vars[0][i]

    # Predict
    K_array, _ = kalman_dot(y[0, :] - np.dot(C, m0), S0, C, R)
    mf[0] = m0 + K_array
    Vf[0, :] = S0 - K_array
    S[0] = np.eye(m0.shape[0])
    innovations[0] = y[0] - np.dot(C, mf[0])
    innovation_cov[0] = np.dot(C, np.dot(S0, C.T)) + R

    # Kalman filter update for subsequent time steps
    for t in range(1, T):
        # Propagate the state
        mf[t, :] = np.dot(A, mf[t - 1, :])
        S[t - 1] = np.dot(A, np.dot(Vf[t - 1, :], A.T)) + Q

        if np.sum(~np.isnan(y[t, :])) >= 2:  # Check if any value in ys[t] is not NaN
            # Update R for time-varying observation variance
            for i in range(ensemble_vars.shape[1]):
                R[i, i] = ensemble_vars[t][i]

            # Update state estimate and covariance matrix
            innovations[t] = y[t, :] - np.dot(C, np.dot(A, mf[t - 1, :]))
            K_array, _ = kalman_dot(innovations[t], S[t - 1], C, R)
            mf[t, :] += K_array
            K_array, innovation_cov[t] = kalman_dot(np.dot(C, S[t - 1]), S[t - 1], C, R)
            Vf[t, :] = S[t - 1] - K_array
        else:
            Vf[t, :] = S[t - 1]
    return mf, Vf, S, innovations, innovation_cov


def backward_pass(y, mf, Vf, S, A):
    """Implements Kalman-smoothing backwards
    Args:
        y: np.ndarray
            shape (samples, n_keypoints)
        mf: np.ndarray
            shape (samples, n_keypoints)
        Vf: np.ndarray
            shape (samples, n_latents, n_latents)
        S: np.ndarray
            shape (samples, n_latents, n_latents)
        A: np.ndarray
            shape (n_latents, n_latents)
        Q: np.ndarray
            shape (n_latents, n_latents)
        C: np.ndarray
            shape (n_keypoints, n_latents)

    Returns:
        ms: np.ndarray
            shape (samples, n_keypoints)
        Vs: np.ndarray
            shape (samples, n_latents, n_latents)
        CV: np.ndarray
            shape (samples, n_latents, n_latents)
    """
    T = y.shape[0]
    ms = mf.copy()
    Vs = Vf.copy()
    CV = np.zeros(shape=(T - 1, mf.shape[1], mf.shape[1]))

    # Last-time smoothed posterior is equal to last-time filtered posterior
    ms[-1, :] = mf[-1, :]
    Vs[-1, :, :] = Vf[-1, :, :]
    # Smoothing steps
    for i in range(T - 2, -1, -1):
        if not np.all(np.isnan(y[i])):  # Check if all values in ys[i] are not NaN
            try:
                J = np.linalg.solve(S[i], np.dot(A, Vf[i])).T
            except np.linalg.LinAlgError:
                # Skip backward pass for this timestep if matrix is singular
                print(f"Warning: Singular Matrix at time step {i}. Skipping backwards pass"
                      f" at this time step.")
                continue

            Vs[i] = Vf[i] + np.dot(J, np.dot(Vs[i + 1] - S[i], J.T))
            ms[i] = mf[i] + np.dot(J, ms[i + 1] - np.dot(A, mf[i]))
            CV[i] = np.dot(Vs[i + 1], J.T)
    return ms, Vs, CV


def kalman_dot(innovation, V, C, R):
    """ Kalman dot product computation """
    innovation_cov = R + np.dot(C, np.dot(V, C.T))
    innovation_cov_inv = np.linalg.solve(innovation_cov, innovation)
    Ks = np.dot(V, np.dot(C.T, innovation_cov_inv))
    return Ks, innovation_cov


def compute_nll(innovations, innovation_covs, epsilon=1e-6):
    """
    Computes the negative log likelihood, which is a likelihood measurement for the
    EKS prediction. This metric is used (minimized) to optimize s.
    """
    T = innovations.shape[0]
    n_coords = innovations.shape[1]
    nll = 0
    nll_values = []
    c = np.log(2 * np.pi) * n_coords  # The Gaussian normalization constant part
    for t in range(T):
        if not np.any(np.isnan(innovations[t])):  # Check if any value in innovations[t] is not NaN
            # Regularize the innovation covariance matrix by adding epsilon to the diagonal
            reg_innovation_cov = innovation_covs[t] + epsilon * np.eye(n_coords)

            # Compute the log determinant of the regularized covariance matrix
            log_det_S = np.log(np.abs(np.linalg.det(reg_innovation_cov)) + epsilon)
            solved_term = np.linalg.solve(reg_innovation_cov, innovations[t])
            quadratic_term = np.dot(innovations[t], solved_term)

            # Compute the NLL increment for time step t
            nll_increment = 0.5 * np.abs((log_det_S + quadratic_term + c))
            nll_values.append(nll_increment)
            nll += nll_increment
    return nll, nll_values


# -------------------------------------------------------------------------------------
# Fast Core Functions: These functions are fast versions used by the singlecam script
# and will eventually replace the Original Core Functions
# -------------------------------------------------------------------------------------

@typechecked
def jax_ensemble(
    marker_array: MarkerArray,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    nan_replacement: float = 1000.0,  # Default replacement value for NaNs
) -> MarkerArray:
    """
    Compute ensemble mean/median and variance of a marker array using JAX.
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


def kalman_filter_step(carry, inputs):
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


def kalman_filter_step_nlls(carry, inputs):
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


# Always run the sequential filter on CPU.
@partial(jit, backend='cpu')
def jax_forward_pass(y, m0, cov0, A, Q, C, R, ensemble_vars):
    """
    Kalman Filter for a single keypoint
    (can be vectorized using vmap for handling multiple keypoints in parallel)
    Parameters:
        y: Shape (num_timepoints, observation_dimension).
        m0: Shape (state_dim,). Initial state of system.
        cov0: Shape (state_dim, state_dim). Initial covariance of state variable.
        A: Shape (state_dim, state_dim). Process transition matrix.
        Q: Shape (state_dim, state_dim). Process noise covariance matrix.
        C: Shape (observation_dim, state_dim). Observation coefficient matrix.
        R: Shape (observation_dim, observation_dim). Observation noise covar matrix.
        ensemble_vars: Shape (num_timepoints, observation_dimension). Time-varying obs noise var.

    Returns:
        mfs: Shape (timepoints, state_dim). Mean filter state at each timepoint.
        Vfs: Shape (timepoints, state_dim, state_dim). Covar for each filtered estimate.
        nll_net: Shape (1,). Negative log likelihood observations -log (p(y_1, ..., y_T))
    """
    # Initialize carry
    carry = (m0, cov0, A, Q, C, 0)

    # Run the scan, passing y and ensemble_vars as inputs to kalman_filter_step
    carry, outputs = jax.lax.scan(kalman_filter_step, carry, (y, ensemble_vars))
    mfs, Vfs, _ = outputs
    nll_net = carry[-1]
    return mfs, Vfs, nll_net


def jax_forward_pass_nlls(y, m0, cov0, A, Q, C, R, ensemble_vars):
    """
    Kalman Filter for a single keypoint
    (can be vectorized using vmap for handling multiple keypoints in parallel)
    Parameters:
        y: Shape (num_timepoints, observation_dimension).
        m0: Shape (state_dim,). Initial state of system.
        cov0: Shape (state_dim, state_dim). Initial covariance of state variable.
        A: Shape (state_dim, state_dim). Process transition matrix.
        Q: Shape (state_dim, state_dim). Process noise covariance matrix.
        C: Shape (observation_dim, state_dim). Observation coefficient matrix.
        R: Shape (observation_dim, observation_dim). Observation noise covar matrix.

    Returns:
        mfs: Shape (timepoints, state_dim). Mean filter state at each timepoint.
        Vfs: Shape (timepoints, state_dim, state_dim). Covar for each filtered estimate.
        nll_net: Shape (1,). Negative log likelihood observations -log (p(y_1, ..., y_T))
        nll_array: Shape (num_timepoints,). Incremental negative log-likelihood at each timepoint.
    """
    # Ensure R is a (2, 2) matrix
    if R.ndim == 1:
        R = jnp.diag(R)

    # Initialize carry
    num_timepoints = y.shape[0]
    nll_array_init = jnp.zeros(num_timepoints)  # Preallocate an array with zeros
    t_init = 1  # Initialize the time step counter
    carry = (m0, cov0, A, Q, C, 0, nll_array_init, t_init)

    # Run the scan, passing y and ensemble_vars
    carry, outputs = jax.lax.scan(kalman_filter_step_nlls, carry, (y, ensemble_vars))
    mfs, Vfs, _ = outputs
    nll_net = carry[-3]  # Total NLL
    nll_array = carry[-2]  # Array of incremental NLL values

    return mfs, Vfs, nll_net, nll_array


def kalman_smoother_step(carry, X):
    m_ahead_smooth, v_ahead_smooth, A, Q = carry
    m_curr_filter, v_curr_filter = X[0], X[1]

    # Compute the smoother gain
    ahead_cov = jnp.dot(A, jnp.dot(v_curr_filter, A.T)) + Q

    smoothing_gain = jsc.linalg.solve(ahead_cov, jnp.dot(A, v_curr_filter.T)).T
    smoothed_state = m_curr_filter + jnp.dot(smoothing_gain, m_ahead_smooth - m_curr_filter)
    smoothed_cov = v_curr_filter + jnp.dot(jnp.dot(smoothing_gain, v_ahead_smooth - ahead_cov),
                                           smoothing_gain.T)

    return (smoothed_state, smoothed_cov, A, Q), (smoothed_state, smoothed_cov)


@partial(jit, backend='cpu')
def jax_backward_pass(mfs, Vfs, A, Q):
    """
    Runs the kalman smoother given the filtered values
    Parameters:
        mfs: Shape (timepoints, state_dim). The kalman-filtered means of the data.
        Vfs: Shape (timepoints, state_dim, state_dimension).
            The kalman-filtered covariance matrix of the state vector at each time point.
        A: Shape (state_dim, state_dim). The process transition matrix
        Q: Shape (state_dim, state_dim). The covariance of the process noise.
    Returns:
        smoothed_states: Shape (timepoints, state_dim).
            The smoothed estimates for the state vector starting at the first timepoint
            where observations are possible.
        smoothed_state_covariances: Shape (timepoints, state_dim, state_dim).
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


def single_timestep_nll(innovation, innovation_cov):
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


def final_forwards_backwards_pass(process_cov, s, ys, m0s, S0s, Cs, As, Rs, ensemble_vars):
    """
    Perform final smoothing with the optimized smoothing parameters.

    Parameters:
        process_cov: Shape (keypoints, state_coords, state_coords). Process noise covariance matrix
        s: Shape (keypoints,). We scale the process noise covariance by this value at each keypoint
        ys: Shape (keypoints, frames, observation_coordinates). Observations for all keypoints.
        m0s: Shape (keypoints, state_coords). Initial ensembled mean state for each keypoint.
        S0s: Shape (keypoints, state_coords, state_coords). Initial ensembled state covars fek.
        Cs: Shape (keypoints, obs_coords, state_coords). Observation measurement coeff matrix.
        As: Shape (keypoints, state_coords, state_coords). Process matrix for each keypoint.
        Rs: Shape (keypoints, obs_coords, obs_coords). Measurement noise covariance.

    Returns:
        smoothed means: Shape (keypoints, timepoints, coords).
            Kalman smoother state estimates outputs for all frames/all keypoints.
        smoothed covariances: Shape (num_keypoints, num_state_coordinates, num_state_coordinates)
    """

    # Initialize
    n_keypoints = ys.shape[0]
    ms_array = []
    Vs_array = []
    nlls_array = []
    Qs = s[:, None, None] * process_cov

    # Run forward and backward pass for each keypoint
    for k in range(n_keypoints):
        mf, Vf, nll, nll_array = jax_forward_pass_nlls(
            ys[k], m0s[k], S0s[k], As[k], Qs[k], Cs[k], Rs[k], ensemble_vars[:, k, :])
        ms, Vs = jax_backward_pass(mf, Vf, As[k], Qs[k])

        ms_array.append(np.array(ms))
        Vs_array.append(np.array(Vs))
        nlls_array.append(np.array(nll_array))

    smoothed_means = np.stack(ms_array, axis=0)
    smoothed_covariances = np.stack(Vs_array, axis=0)

    return smoothed_means, smoothed_covariances, nlls_array

# -------------------------------------------------------------------------------------
# Optimization: These functions are related to optimizing the smoothing hyperparameter
# -------------------------------------------------------------------------------------


def compute_initial_guesses(ensemble_vars):
    """Computes an initial guess for optimized s as the stdev of temporal differences."""
    # Consider only the first 2000 entries in ensemble_vars
    ensemble_vars = ensemble_vars[:2000]

    # Compute ensemble mean
    ensemble_mean = np.nanmean(ensemble_vars, axis=0)
    if ensemble_mean is None:
        raise ValueError("No data found. Unable to compute ensemble mean.")

    # Initialize list to store temporal differences
    temporal_diffs_list = []

    # Iterate over each time step
    for i in range(1, len(ensemble_mean)):
        # Compute temporal difference for current time step
        temporal_diff = ensemble_mean - ensemble_vars[i]
        temporal_diffs_list.append(temporal_diff)

    # Compute standard deviation of temporal differences
    std_dev_guess = round(np.std(temporal_diffs_list), 5)
    return std_dev_guess


def optimize_smooth_param(
    cov_mats: np.ndarray,
    ys: np.ndarray,
    m0s: np.ndarray,
    S0s: np.ndarray,
    Cs: np.ndarray,
    As: np.ndarray,
    Rs: np.ndarray,
    ensemble_vars: np.ndarray,
    s_frames: list | None,
    smooth_param: float | list | None,
    blocks: list = [],
    maxiter: int = 1000,
    verbose: bool = False,
) -> tuple:
    """Optimize smoothing parameter, and use the result to run the kalman filter-smoother.

    Parameters:
        cov_mats: Covariance matrices.
        ys: Observations. Shape (keypoints, frames, coordinates). coordinate is usually 2
        m0s: Initial mean state.
        S0s: Initial state covariance.
        Cs: Measurement function.
        As: State-transition matrix.
        Rs: Measurement noise covariance.
        ensemble_vars: Ensemble variances.
        s_frames: List of frames.
        smooth_param: Smoothing parameter.
        blocks: keypoints to be blocked for correlated noise. Generates on smoothing param per
            block, as opposed to per keypoint.
            Specified by the form "x1, x2, x3; y1, y2" referring to keypoint indices (start at 0)
        maxiter
        verbose

    Returns:
        tuple: Final smoothing parameters, smoothed means, smoothed covariances,
               negative log-likelihoods, negative log-likelihood values.

    """

    n_keypoints = ys.shape[0]
    s_finals = []
    if len(blocks) == 0:
        for n in range(n_keypoints):
            blocks.append([n])
    if verbose:
        print(f'Correlated keypoint blocks: {blocks}')

    @partial(jit)
    def nll_loss_sequential_scan(s, cov_mats, cropped_ys, m0s, S0s, Cs, As, Rs, ensemble_vars):
        s = jnp.exp(s)  # To ensure positivity
        return smooth_min(
            s, cov_mats, cropped_ys, m0s, S0s, Cs, As, Rs, ensemble_vars)

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
            Rs_crop = Rs[selector]
            y_subset = cropped_ys[selector]
            ensemble_vars_crop = np.swapaxes(ensemble_vars[:, selector, :], 0, 1)

            def step(s, opt_state):
                loss, grads = jax.value_and_grad(loss_function)(
                    s, cov_mats_sub, y_subset, m0s_crop, S0s_crop, Cs_crop, As_crop, Rs_crop,
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
    ms, Vs, nlls = final_forwards_backwards_pass(
        cov_mats, s_finals, ys, m0s, S0s, Cs, As, Rs, ensemble_vars,
    )

    return s_finals, ms, Vs, nlls


def inner_smooth_min_routine(y, m0, S0, A, Q, C, R, ensemble_var):
    # Run filtering with the current smooth_param
    _, _, nll = jax_forward_pass(y, m0, S0, A, Q, C, R, ensemble_var)
    return nll


inner_smooth_min_routine_vmap = vmap(inner_smooth_min_routine, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))


def smooth_min(smooth_param, cov_mats, ys, m0s, S0s, Cs, As, Rs, ensemble_vars):
    """
    Smooths once using the given smooth_param. Returns only the nll, which is the parameter to
    be minimized using the scipy.minimize() function.

    Parameters:
    smooth_param (float): Smoothing parameter.
    block (list): List of blocks.
    cov_mats (np.ndarray): Covariance matrices.
    ys (np.ndarray): Observations.
    m0s (np.ndarray): Initial mean state.
    S0s (np.ndarray): Initial state covariance.
    Cs (np.ndarray): Measurement function.
    As (np.ndarray): State-transition matrix.
    Rs (np.ndarray): Measurement noise covariance.

    Returns:
    float: Negative log-likelihood.
    """
    # Adjust Q based on smooth_param and cov_matrix
    Qs = smooth_param * cov_mats
    nlls = jnp.sum(inner_smooth_min_routine_vmap(ys, m0s, S0s, As, Qs, Cs, Rs, ensemble_vars))
    return nlls

# -------------------------------------------------------------------------------------
# Misc: These miscellaneous functions generally have specific computations used by the
# core functions or the smoothers
# -------------------------------------------------------------------------------------


def eks_zscore(eks_predictions, ensemble_means, ensemble_vars, min_ensemble_std=1e-5):
    """Computes zscore between eks prediction and the ensemble for a single keypoint.
    Args:
        eks_predictions: list
            EKS prediction for each coordinate (x and ys) for as single keypoint - (samples, 2)
        ensemble_means: list
            Ensemble mean for each coordinate (x and ys) for as single keypoint - (samples, 2)
        ensemble_vars: string
            Ensemble var for each coordinate (x and ys) for as single keypoint - (samples, 2)
        min_ensemble_std:
            Minimum std threshold to reduce the effect of low ensemble std (default 1e-5).
    Returns:
        z_score
            z_score for each time point - (samples, 1)
    """
    ensemble_std = np.sqrt(
        # trace of covariance matrix - multi-d variance measure
        ensemble_vars[:, 0] + ensemble_vars[:, 1])
    num = np.sqrt(
        (eks_predictions[:, 0]
         - ensemble_means[:, 0]) ** 2
        + (eks_predictions[:, 1] - ensemble_means[:, 1]) ** 2)
    thresh_ensemble_std = ensemble_std.copy()
    thresh_ensemble_std[thresh_ensemble_std < min_ensemble_std] = min_ensemble_std
    z_score = num / thresh_ensemble_std
    return z_score, ensemble_std


def compute_covariance_matrix(ensemble_preds):
    """Compute the covariance matrix E for correlated noise dynamics.

    Args:
        ensemble_preds: shape (T, n_keypoints, n_coords) containing the ensemble predictions.

    Returns:
        E: A 2K x 2K covariance matrix where K is the number of keypoints.

    """
    # Get the number of time steps, keypoints, and coordinates
    T, n_keypoints, n_coords = ensemble_preds.shape

    # Flatten the ensemble predictions to shape (T, 2K) where K is the number of keypoints
    # flattened_preds = ensemble_preds.reshape(T, -1)

    # Compute the temporal differences
    # temporal_diffs = np.diff(flattened_preds, axis=0)

    # Compute the covariance matrix of the temporal differences
    # E = np.cov(temporal_diffs, rowvar=False)

    # Index covariance matrix into blocks for each keypoint
    cov_mats = []
    for i in range(n_keypoints):
        # E_block = extract_submatrix(E, i)  -- using E_block instead of the identity matrix
        # leads to a correlated dynamics model, but further debugging required due to negative vars
        cov_mats.append([[1, 0], [0, 1]])
    cov_mats = jnp.array(cov_mats)
    return cov_mats


def extract_submatrix(Qs, i, submatrix_size=2):
    # Compute the start indices for the submatrix
    i_q = 2 * i
    start_indices = (i_q, i_q)

    # Use jax.lax.dynamic_slice to extract the submatrix
    submatrix = jax.lax.dynamic_slice(Qs, start_indices, (submatrix_size, submatrix_size))

    return submatrix


def center_predictions(
    ensemble_marker_array: MarkerArray,
    quantile_keep_pca: float
):
    """
    Filter frames based on variance, compute mean coordinates, and scale predictions.

    Args:
        ensemble_marker_array: Ensemble MarkerArray containing predicted positions and variances.
        quantile_keep_pca: Threshold percentage for filtering low-variance frames.

    Returns:
        tuple:
            valid_frames_mask (np.ndarray): Boolean mask of valid frames per keypoint.
            emA_centered_preds (MarkerArray): Centered ensemble predictions.
            emA_good_centered_preds (MarkerArray): Centered ensemble predictions for valid frames.
            emA_means (MarkerArray): Mean x and y coords for each camera.
    """
    n_models, n_cameras, n_frames, n_keypoints, _ = ensemble_marker_array.shape
    assert n_models == 1, "MarkerArray should have n_models = 1 after ensembling."

    emA_preds = ensemble_marker_array.slice_fields("x", "y")
    emA_vars = ensemble_marker_array.slice_fields("var_x", "var_y")

    # Maximum variance for each keypoint in each frame, independent of camera
    max_vars_per_frame = np.max(emA_vars.array, axis=(0, 1, 4))  # Shape: (n_frames, n_keypoints)
    # Compute variance threshold for each keypoint
    thresholds = np.percentile(max_vars_per_frame, quantile_keep_pca, axis=0)

    valid_frames_mask = max_vars_per_frame <= thresholds  # Shape: (n_frames, n_keypoints)

    min_frames = float('inf')  # Initialize min_frames to infinity

    emA_centered_preds_list = []
    emA_good_centered_preds_list = []
    emA_means_list = []
    good_frame_indices_list = []

    for k in range(n_keypoints):
        # Find valid frame indices for the current keypoint
        good_frame_indices = np.where(valid_frames_mask[:, k])[0]  # Shape: (n_filtered_frames,)

        # Update min_frames to track the minimum number of valid frames across keypoints
        if len(good_frame_indices) < min_frames:
            min_frames = len(good_frame_indices)

        good_frame_indices_list.append(good_frame_indices)

    # Now, reprocess each keypoint using only `min_frames` frames
    for k in range(n_keypoints):
        good_frame_indices = good_frame_indices_list[k][:min_frames]  # Truncate to min_frames

        # Extract valid frames for this keypoint
        good_preds_k = emA_preds.array[:, :, good_frame_indices, k, :]
        good_preds_k = np.expand_dims(good_preds_k, axis=3)

        # Scale predictions by subtracting means (over frames) from predictions
        means_k = np.mean(good_preds_k, axis=2)[:, :, None, :, :]
        centered_preds_k = emA_preds.slice("keypoints", k).array - means_k
        good_centered_preds_k = good_preds_k - means_k

        emA_centered_preds_list.append(
            MarkerArray(centered_preds_k, data_fields=["x", "y"]))
        emA_good_centered_preds_list.append(
            MarkerArray(good_centered_preds_k, data_fields=["x", "y"]))
        emA_means_list.append(MarkerArray(means_k, data_fields=["x", "y"]))

    # Concatenate all keypoint-wise filtered results along the keypoints axis
    emA_centered_preds = MarkerArray.stack(emA_centered_preds_list, "keypoints")
    emA_good_centered_preds = MarkerArray.stack(emA_good_centered_preds_list, "keypoints")
    emA_means = MarkerArray.stack(emA_means_list, "keypoints")

    return valid_frames_mask, emA_centered_preds, emA_good_centered_preds, emA_means
