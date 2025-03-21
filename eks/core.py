from functools import partial

import jax
import jax.scipy as jsc
import numpy as np
from jax import jit
from jax import numpy as jnp
from typeguard import typechecked

from eks.marker_array import MarkerArray

# ------------------------------------------------------------------------------------------
# Original Core Functions: These functions are still in use for the multicam and IBL scripts
# as of this update, but will eventually be replaced the with faster versions used in
# the singlecam script
# ------------------------------------------------------------------------------------------


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

# ----- Sequential Functions for CPU -----

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


def kalman_filter_step(carry, curr_y):
    m_prev, V_prev, A, Q, C, R, nll_net = carry

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

    return (m_t, V_t, A, Q, C, R, nll_net), (m_t, V_t, nll_current)


def kalman_filter_step_nlls(carry, inputs):
    # Unpack carry and inputs
    m_prev, V_prev, A, Q, C, R, nll_net, nll_array, t = carry
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
    return (m_t, V_t, A, Q, C, R, nll_net, nll_array, t), (m_t, V_t, nll_current)


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
    carry = (m0, cov0, A, Q, C, R, 0)

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
    carry = (m0, cov0, A, Q, C, R, 0, nll_array_init, t_init)

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


def compute_initial_guesses(ensemble_vars):
    """Computes an initial guess for optimized s, which is the stdev of temporal differences."""
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
