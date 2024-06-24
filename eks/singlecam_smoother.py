import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import jit, vmap

from eks.core import (
    compute_covariance_matrix,
    compute_initial_guesses,
    eks_zscore,
    jax_backward_pass,
    jax_ensemble,
    jax_forward_pass,
    pkf_and_loss,
)
from eks.utils import crop_frames, make_dlc_pandas_index


def ensemble_kalman_smoother_singlecam(
        markers_3d_array, bodypart_list, smooth_param, s_frames, blocks=[],
        ensembling_mode='median',
        zscore_threshold=2):
    """
    Perform Ensemble Kalman Smoothing on 3D marker data from a single camera.

    Parameters:
    markers_3d_array (np.ndarray): 3D array of marker data.
    bodypart_list (list): List of body parts.
    smooth_param (float): Smoothing parameter.
    s_frames (list): List of frames.
    ensembling_mode (str): Mode for ensembling ('median' by default).
    zscore_threshold (float): Z-score threshold.

    Returns:
    tuple: Dataframes with smoothed predictions, final smoothing parameters, NLL values.
    """

    T = markers_3d_array.shape[1]
    n_keypoints = markers_3d_array.shape[2] // 3
    n_coords = 2

    # Compute ensemble statistics
    print("Ensembling models")
    ensemble_preds, ensemble_vars, keypoints_avg_dict = jax_ensemble(
        markers_3d_array, mode=ensembling_mode)

    # Calculate mean and adjusted observations
    mean_obs_dict, adjusted_obs_dict, scaled_ensemble_preds = adjust_observations(
        keypoints_avg_dict, n_keypoints, ensemble_preds.copy())

    # Initialize Kalman filter values
    m0s, S0s, As, cov_mats, Cs, Rs, ys = initialize_kalman_filter(
        scaled_ensemble_preds, adjusted_obs_dict, n_keypoints)

    # Main smoothing function
    s_finals, ms, Vs = singlecam_optimize_smooth(
        cov_mats, ys, m0s, S0s, Cs, As, Rs, ensemble_vars,
        s_frames, smooth_param, blocks)

    y_m_smooths = np.zeros((n_keypoints, T, n_coords))
    y_v_smooths = np.zeros((n_keypoints, T, n_coords, n_coords))
    eks_preds_array = np.zeros(y_m_smooths.shape)
    dfs = []
    df_dicts = []

    # Process each keypoint
    for k in range(n_keypoints):
        y_m_smooths[k] = np.dot(Cs[k], ms[k].T).T
        y_v_smooths[k] = np.swapaxes(np.dot(Cs[k], np.dot(Vs[k], Cs[k].T)), 0, 1)
        mean_x_obs = mean_obs_dict[3 * k]
        mean_y_obs = mean_obs_dict[3 * k + 1]

        # Computing z-score
        eks_preds_array[k] = y_m_smooths[k].copy()
        eks_preds_array[k] = np.asarray([eks_preds_array[k].T[0] + mean_x_obs,
                                         eks_preds_array[k].T[1] + mean_y_obs]).T
        zscore = eks_zscore(eks_preds_array[k],
                            ensemble_preds[:, k, :],
                            ensemble_vars[:, k, :],
                            min_ensemble_std=zscore_threshold)

        # Final Cleanup
        pdindex = make_dlc_pandas_index([bodypart_list[k]],
                                        labels=["x", "y", "likelihood", "x_var", "y_var",
                                                "zscore"])
        var = np.empty(y_m_smooths[k].T[0].shape)
        var[:] = np.nan
        pred_arr = np.vstack([
            y_m_smooths[k].T[0] + mean_x_obs,
            y_m_smooths[k].T[1] + mean_y_obs,
            var,
            y_v_smooths[k][:, 0, 0],
            y_v_smooths[k][:, 1, 1],
            zscore,
        ]).T
        df = pd.DataFrame(pred_arr, columns=pdindex)
        dfs.append(df)
        df_dicts.append({bodypart_list[k] + '_df': df})

    return df_dicts, s_finals


def adjust_observations(keypoints_avg_dict, n_keypoints, scaled_ensemble_preds):
    """
    Adjust observations by computing mean and adjusted observations for each keypoint.

    Parameters:
    keypoints_avg_dict (dict): Dictionary of keypoints averages.
    n_keypoints (int): Number of keypoints.
    scaled_ensemble_preds (np.ndarray): Scaled ensemble predictions.

    Returns:
    tuple: Mean observations dictionary, adjusted observations dictionary, scaled ensemble preds.
    """

    # Convert dictionaries to JAX arrays
    keypoints_avg_array = jnp.array([keypoints_avg_dict[k] for k in keypoints_avg_dict.keys()])
    x_keys = jnp.array([3 * i for i in range(n_keypoints)])
    y_keys = jnp.array([3 * i + 1 for i in range(n_keypoints)])

    def compute_adjusted_means(i):
        mean_x_obs = jnp.nanmean(keypoints_avg_array[2 * i])
        mean_y_obs = jnp.nanmean(keypoints_avg_array[2 * i + 1])
        adjusted_x_obs = keypoints_avg_array[2 * i] - mean_x_obs
        adjusted_y_obs = keypoints_avg_array[2 * i + 1] - mean_y_obs
        return mean_x_obs, mean_y_obs, adjusted_x_obs, adjusted_y_obs

    means_and_adjustments = jax.vmap(compute_adjusted_means)(jnp.arange(n_keypoints))

    mean_x_obs, mean_y_obs, adjusted_x_obs, adjusted_y_obs = means_and_adjustments

    # Convert JAX arrays to NumPy arrays for dictionary keys
    x_keys_np = np.array(x_keys)
    y_keys_np = np.array(y_keys)

    mean_obs_dict = {x_keys_np[i]: mean_x_obs[i] for i in range(n_keypoints)}
    mean_obs_dict.update({y_keys_np[i]: mean_y_obs[i] for i in range(n_keypoints)})

    adjusted_obs_dict = {x_keys_np[i]: adjusted_x_obs[i] for i in range(n_keypoints)}
    adjusted_obs_dict.update({y_keys_np[i]: adjusted_y_obs[i] for i in range(n_keypoints)})

    # Ensure scaled_ensemble_preds is a JAX array
    scaled_ensemble_preds = jnp.array(scaled_ensemble_preds)

    def scale_ensemble_preds(mean_x_obs, mean_y_obs, scaled_ensemble_preds, i):
        scaled_ensemble_preds = scaled_ensemble_preds.at[:, i, 0].add(-mean_x_obs)
        scaled_ensemble_preds = scaled_ensemble_preds.at[:, i, 1].add(-mean_y_obs)
        return scaled_ensemble_preds

    for i in range(n_keypoints):
        mean_x = mean_obs_dict[x_keys_np[i]]
        mean_y = mean_obs_dict[y_keys_np[i]]
        scaled_ensemble_preds = scale_ensemble_preds(mean_x, mean_y, scaled_ensemble_preds, i)

    return mean_obs_dict, adjusted_obs_dict, scaled_ensemble_preds


def initialize_kalman_filter(scaled_ensemble_preds, adjusted_obs_dict, n_keypoints):
    """
    Initialize the Kalman filter values.

    Parameters:
    scaled_ensemble_preds (np.ndarray): Scaled ensemble predictions.
    adjusted_obs_dict (dict): Adjusted observations dictionary.
    n_keypoints (int): Number of keypoints.

    Returns:
    tuple: Initial Kalman filter values and covariance matrices.
    """

    # Convert inputs to JAX arrays
    scaled_ensemble_preds = jnp.array(scaled_ensemble_preds)

    # Extract the necessary values from adjusted_obs_dict
    adjusted_x_obs_list = [adjusted_obs_dict[3 * i] for i in range(n_keypoints)]
    adjusted_y_obs_list = [adjusted_obs_dict[3 * i + 1] for i in range(n_keypoints)]

    # Convert these lists to JAX arrays
    adjusted_x_obs_array = jnp.array(adjusted_x_obs_list)
    adjusted_y_obs_array = jnp.array(adjusted_y_obs_list)

    def init_kalman(i, adjusted_x_obs, adjusted_y_obs):
        m0 = jnp.array([0.0, 0.0])  # initial state: mean
        S0 = jnp.array([[jnp.nanvar(adjusted_x_obs), 0.0],
                        [0.0, jnp.nanvar(adjusted_y_obs)]])  # diagonal: var
        A = jnp.array([[1.0, 0], [0, 1.0]])  # state-transition matrix
        C = jnp.array([[1, 0], [0, 1]])  # Measurement function
        R = jnp.eye(2)  # placeholder diagonal matrix for ensemble variance
        y_obs = scaled_ensemble_preds[:, i, :]

        return m0, S0, A, C, R, y_obs

    # Use vmap to vectorize the initialization over all keypoints
    init_kalman_vmap = jax.vmap(init_kalman, in_axes=(0, 0, 0))
    m0s, S0s, As, Cs, Rs, y_obs_array = init_kalman_vmap(jnp.arange(n_keypoints),
                                                         adjusted_x_obs_array,
                                                         adjusted_y_obs_array)
    cov_mats = compute_covariance_matrix(scaled_ensemble_preds)
    return m0s, S0s, As, cov_mats, Cs, Rs, y_obs_array


def singlecam_optimize_smooth(
        cov_mats, ys, m0s, S0s, Cs, As, Rs, ensemble_vars,
        s_frames, smooth_param, blocks=[], maxiter=1000):
    """
    Optimize smoothing parameter, and use the result to run the kalman filter-smoother

    Parameters:
    cov_mats (np.ndarray): Covariance matrices.
    ys (np.ndarray): Observations. Shape (keypoints, frames, coordinates). coordinate is usually 2
    m0s (np.ndarray): Initial mean state.
    S0s (np.ndarray): Initial state covariance.
    Cs (np.ndarray): Measurement function.
    As (np.ndarray): State-transition matrix.
    Rs (np.ndarray): Measurement noise covariance.
    ensemble_vars (np.ndarray): Ensemble variances.
    s_frames (list): List of frames.
    smooth_param (float): Smoothing parameter.
    blocks (list): List of blocks.

    Returns:
    tuple: Final smoothing parameters, smoothed means, smoothed covariances,
    negative log-likelihoods, negative log-likelihood values.
    """

    n_keypoints = ys.shape[0]
    s_finals = []
    if blocks == []:
        for n in range(n_keypoints):
            blocks.append([n])
    print(f'Correlated keypoint blocks: {blocks}')

    # Depending on whether we use GPU, choose parallel or sequential smoothing param optimization
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        print("Using GPU")

        @partial(jit)
        def nll_loss_parallel_scan(s, cov_mats, cropped_ys, m0s, S0s, Cs, As, Rs):
            s = jnp.exp(s)  # To ensure positivity
            output = singlecam_smooth_min_parallel(s, cov_mats, cropped_ys, m0s, S0s, Cs, As, Rs)
            return output

        loss_function = nll_loss_parallel_scan
    except:
        print("Using CPU")

        @partial(jit)
        def nll_loss_sequential_scan(s, cov_mats, cropped_ys, m0s, S0s, Cs, As, Rs):
            s = jnp.exp(s)  # To ensure positivity
            return singlecam_smooth_min(s, cov_mats, cropped_ys, m0s, S0s, Cs, As, Rs)

        loss_function = nll_loss_sequential_scan

    # Optimize smooth_param
    if smooth_param is not None:
        s_finals = [smooth_param]
    else:
        guesses = []
        cropped_ys = []
        for k in range(n_keypoints):
            current_guess = compute_initial_guesses(ensemble_vars[:, k, :])
            guesses.append(current_guess)
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

            def step(s, opt_state):
                loss, grads = jax.value_and_grad(loss_function)(s, cov_mats_sub, y_subset,
                                                                m0s_crop,
                                                                S0s_crop, Cs_crop, As_crop,
                                                                Rs_crop)
                updates, opt_state = optimizer.update(grads, opt_state)
                s = optax.apply_updates(s, updates)
                return s, opt_state, loss

            prev_loss = jnp.inf
            for iteration in range(maxiter):
                start_time = time.time()
                s_init, opt_state, loss = step(s_init, opt_state)

                # if iteration % 10 == 0 or iteration == maxiter - 1:
                #     print(f'Iteration {iteration}, Current loss: {loss}, Current s: {s_init}')

                tol = 0.001 * jnp.abs(jnp.log(prev_loss))
                if jnp.linalg.norm(loss - prev_loss) < tol + 1e-6:
                #    print(
                #        f'Converged at iteration {iteration} with '
                #        f'smoothing parameter {jnp.exp(s_init)}. NLL={loss}')
                    break

                prev_loss = loss

            s_final = jnp.exp(s_init)  # Convert back from log-space
            for b in block:
                print(f's={s_final} for keypoint {b}')
                s_finals.append(s_final)

    s_finals = np.array(s_finals)
    # Final smooth with optimized s
    ms, Vs = final_forwards_backwards_pass(
        cov_mats, s_finals,
        ys, m0s, S0s, Cs, As, Rs)

    return s_finals, ms, Vs


######
## Routines that use the sequential kalman filter implementation to arrive at the NLL function
## Note: this code is set up to always run on CPU.
######

def inner_smooth_min_routine(y, m0, S0, A, Q, C, R):
    # Run filtering with the current smooth_param
    _, _, nll = jax_forward_pass(y, m0, S0, A, Q, C, R)
    return nll


inner_smooth_min_routine_vmap = vmap(inner_smooth_min_routine, in_axes=(0, 0, 0, 0, 0, 0, 0))


def singlecam_smooth_min(
        smooth_param, cov_mats, ys, m0s, S0s, Cs, As, Rs):
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
    nlls = jnp.sum(inner_smooth_min_routine_vmap(ys, m0s, S0s, As, Qs, Cs, Rs))
    return nlls


def inner_smooth_min_routine_parallel(y, m0, S0, A, Q, C, R):
    # Run filtering with the current smooth_param
    means, covariances, NLL = pkf_and_loss(y, m0, S0, A, Q, C, R)
    return jnp.sum(NLL)


inner_smooth_min_routine_parallel_vmap = jit(
    vmap(inner_smooth_min_routine_parallel, in_axes=(0, 0, 0, 0, 0, 0, 0)))


# ------------------------------------------------------------------------------------------------
# Routines that use the parallel scan kalman filter implementation to arrive at the NLL function.
# Note: This should only be run on GPUs
# ------------------------------------------------------------------------------------------------


def singlecam_smooth_min_parallel(
        smooth_param, cov_mats, observations, initial_means, initial_covariances, Cs, As, Rs):
    """
    Computes the maximum likelihood estimator for the process noise variance (smoothness param).
    This function is parallelized to process all keypoints in a given block.
    KEY: This function uses the parallel scan algorithm, which has effectively O(log(n))
    runtime on GPUs. On CPUs, it is slower than the jax.lax.scan implementation above.

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
    values = inner_smooth_min_routine_parallel_vmap(observations, initial_means,
                                                    initial_covariances, As, Qs, Cs, Rs)
    return jnp.sum(values)


def final_forwards_backwards_pass(process_cov, s, ys, m0s, S0s, Cs, As, Rs):
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
    Qs = s[:, None, None] * process_cov

    # Run forward and backward pass for each keypoint
    for k in range(n_keypoints):
        mf, Vf, nll = jax_forward_pass(ys[k], m0s[k], S0s[k], As[k], Qs[k], Cs[k], Rs[k])
        ms, Vs = jax_backward_pass(mf, Vf, As[k], Qs[k])
        ms_array.append(np.array(ms))
        Vs_array.append(np.array(Vs))

    smoothed_means = np.stack(ms_array, axis=0)
    smoothed_covariances = np.stack(Vs_array, axis=0)

    return smoothed_means, smoothed_covariances
