import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import jit, vmap
from typeguard import typechecked

from eks.core import (
    compute_initial_guesses,
    jax_backward_pass,
    jax_ensemble,
    jax_forward_pass,
    jax_forward_pass_nlls,
)
from eks.utils import crop_frames, format_data, make_dlc_pandas_index
from eks.marker_array import MarkerArray, input_dfs_to_markerArray
from eks.multicam_smoother import scale_predictions

@typechecked
def fit_eks_singlecam(
    input_source: str | list,
    save_file: str,
    bodypart_list: list | None = None,
    smooth_param: float | list | None = None,
    s_frames: list | None = None,
    blocks: list = [],
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
) -> tuple:
    """Fit the Ensemble Kalman Smoother for single-camera data.

    Args:
        input_source: directory path or list of CSV file paths. If a directory path, all files
            within this directory will be used.
        save_file: File to save output dataframe.
        bodypart_list: list of body parts to analyze.
        smooth_param: value in (0, Inf); smaller values lead to more smoothing
        s_frames: Frames for automatic optimization if smooth_param is not provided.
        blocks: keypoints to be blocked for correlated noise. Generates on smoothing param per
            block, as opposed to per keypoint.
            Specified by the form "x1, x2, x3; y1, y2" referring to keypoint indices (start at 0)
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        verbose: Extra print statements if True

    Returns:
        tuple:
            df_smoothed (pd.DataFrame)
            s_finals (list): List of optimized smoothing factors for each keypoint.
            input_dfs (list): List of input DataFrames for plotting.
            bodypart_list (list): List of body parts used.

    """
    # Load and format input files using the unified format_data function
    input_dfs_list, keypoint_names = format_data(input_source)

    if bodypart_list is None:
        bodypart_list = keypoint_names
        print(f'Input data loaded for keypoints:\n{bodypart_list}')

    marker_array = input_dfs_to_markerArray([input_dfs_list], bodypart_list, [""])
    # Run the ensemble Kalman smoother
    df_smoothed, smooth_params_final = ensemble_kalman_smoother_singlecam(
        marker_array=marker_array,
        keypoint_names=bodypart_list,
        smooth_param=smooth_param,
        s_frames=s_frames,
        blocks=blocks,
        avg_mode=avg_mode,
        var_mode=var_mode,
        verbose=verbose
    )

    # Save the output DataFrame to CSV
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    df_smoothed.to_csv(save_file)
    print("DataFrames successfully converted to CSV")

    return df_smoothed, smooth_params_final, input_dfs_list, bodypart_list


@typechecked
def ensemble_kalman_smoother_singlecam(
    marker_array: MarkerArray,
    keypoint_names: list,
    smooth_param: float | list | None = None,
    s_frames: list | None = None,
    blocks: list = [],
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
) -> tuple:
    """Perform Ensemble Kalman Smoothing for single-camera data.

    Args:
        marker_array: MarkerArray object containing marker data.
            Shape (n_models, n_cameras, n_frames, n_keypoints, 3 (for x, y, likelihood))
        keypoint_names: List of body parts to run smoothing on
        smooth_param: value in (0, Inf); smaller values lead to more smoothing
        s_frames: List of frames for automatic computation of smoothing parameter
        blocks: keypoints to be blocked for correlated noise. Generates on smoothing param per
            block, as opposed to per keypoint.
            Specified by the form "x1, x2, x3; y1, y2" referring to keypoint indices (start at 0)
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        verbose: True to print out details

    Returns:
        tuple: Dataframes with smoothed predictions, final smoothing parameters.

    """

    n_models, n_cameras, n_frames, n_keypoints, n_data_fields = marker_array.shape

    # MarkerArray (1, 1, n_frames, n_keypoints, 5 (x, y, var_x, var_y, likelihood))
    ensemble_marker_array = jax_ensemble(marker_array, avg_mode=avg_mode, var_mode=var_mode)

    # Save ensemble medians for output
    emA_medians = MarkerArray(
        marker_array=ensemble_marker_array.slice_fields("x", "y"),
        data_fields=["x_median", "y_median"])

    _, emA_scaled_preds, _, emA_means = scale_predictions(
        ensemble_marker_array, quantile_keep_pca=100)  # Should we filter by variance?

    (
        m0s, S0s, As, cov_mats, Cs, Rs
    ) = initialize_kalman_filter(emA_scaled_preds)

    # MarkerArray data_fields=["x", "y", "likelihood", "var_x", "var_y"]
    ensemble_marker_array = MarkerArray.stack_fields(
        emA_scaled_preds,
        ensemble_marker_array.slice_fields("likelihood", "var_x", "var_y"))

    # Prepare params for singlecam_optimize_smooth()
    ensemble_vars = ensemble_marker_array.slice_fields("var_x", "var_y").get_array(squeeze=True)
    ys = ensemble_marker_array.slice_fields("x", "y").get_array(squeeze=True)
    ys = ys.transpose(1, 0, 2)

    # Main smoothing function
    s_finals, ms, Vs, nlls = singlecam_optimize_smooth(
        cov_mats, ys, m0s, S0s, Cs, As, Rs, ensemble_vars,
        s_frames, smooth_param, blocks, verbose)

    y_m_smooths = np.zeros((n_keypoints, n_frames, 2))
    y_v_smooths = np.zeros((n_keypoints, n_frames, 2, 2))

    # Make emAs for smoothed preds and posterior variances -- TODO: refactor into a function
    emA_smoothed_preds_list = []
    emA_postvars_list = []
    for k in range(n_keypoints):
        y_m_smooths[k] = np.dot(Cs[k], ms[k].T).T
        y_v_smooths[k] = np.swapaxes(np.dot(Cs[k], np.dot(Vs[k], Cs[k].T)), 0, 1)
        mean_x_obs = emA_means.slice("keypoints", k).slice_fields("x").get_array(squeeze=True)
        mean_y_obs = emA_means.slice("keypoints", k).slice_fields("y").get_array(squeeze=True)

        # Unscale (re-add means to) smoothed x and y
        smoothed_xs_k = y_m_smooths[k].T[0] + mean_x_obs
        smoothed_ys_k = y_m_smooths[k].T[1] + mean_y_obs

        # Reshape into MarkerArray format
        smoothed_xs_k = smoothed_xs_k[None, None, :, None, None]
        smoothed_ys_k = smoothed_ys_k[None, None, :, None, None]

        # Create smoothed preds emA for current keypoint
        emA_smoothed_xs_k = MarkerArray(smoothed_xs_k, data_fields=["x"])
        emA_smoothed_ys_k = MarkerArray(smoothed_ys_k, data_fields=["y"])
        emA_smoothed_preds_k = MarkerArray.stack_fields(emA_smoothed_xs_k, emA_smoothed_ys_k)
        emA_smoothed_preds_list.append(emA_smoothed_preds_k)

        # Create posterior variance emA for current keypoint
        postvar_xs_k = y_v_smooths[k][:, 0, 0]
        postvar_ys_k = y_v_smooths[k][:, 1, 1]
        postvar_xs_k = postvar_xs_k[None, None, :, None, None]
        postvar_ys_k = postvar_ys_k[None, None, :, None, None]
        emA_postvar_xs_k = MarkerArray(postvar_xs_k, data_fields=["postvar_x"])
        emA_postvar_ys_k = MarkerArray(postvar_ys_k, data_fields=["postvar_y"])
        emA_postvars_k = MarkerArray.stack_fields(emA_postvar_xs_k, emA_postvar_ys_k)
        emA_postvars_list.append(emA_postvars_k)

    emA_smoothed_preds = MarkerArray.stack(emA_smoothed_preds_list, "keypoints")
    emA_postvars = MarkerArray.stack(emA_postvars_list, "keypoints")

    # Create Final MarkerArray
    emA_final = MarkerArray.stack_fields(
        emA_smoothed_preds,  # x, y
        ensemble_marker_array.slice_fields("likelihood"), # likelihood
        emA_medians, # x_median, y_median
        ensemble_marker_array.slice_fields("var_x", "var_y"), # var_x, var_y
        emA_postvars  # postvar_x, postvar_y
    )

    labels = [
        'x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
        'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var'
    ]

    final_array = emA_final.get_array(squeeze=True)

    # Put data into dataframe
    pdindex = make_dlc_pandas_index(keypoint_names, labels=labels)
    final_array = final_array.reshape(n_frames, n_keypoints * len(labels))
    markers_df = pd.DataFrame(final_array, columns=pdindex)

    return markers_df, s_finals


def initialize_kalman_filter(
    emA_scaled_preds: MarkerArray,
) -> tuple:
    """
    Initialize the Kalman filter values.

    Parameters:
        scaled_ensemble_preds (MarkerArray): Scaled ensemble predictions.

    Returns:
        tuple: Initial Kalman filter values and covariance matrices.
    """
    _, _, _, n_keypoints, _ = emA_scaled_preds.shape

    # Shape: (n_frames, n_keypoints, 2 (for x, y))
    scaled_preds = emA_scaled_preds.slice_fields("x", "y").get_array(squeeze=True)

    m0s = np.zeros((n_keypoints, 2))  # Initial state means: (n_keypoints, 2)
    S0s = np.array([
        [[np.nanvar(scaled_preds[:, k, 0]), 0.0],  # [var(x)  0 ]
         [0.0, np.nanvar(scaled_preds[:, k, 1])]]  # [ 0  var(y)]
        for k in range(n_keypoints)
    ])  # Initial covariance matrices: (n_keypoints, 2, 2)

    # State-transition and measurement matrices
    As = np.tile(np.eye(2), (n_keypoints, 1, 1))  # (n_keypoints, 2, 2)
    Cs = np.tile(np.eye(2), (n_keypoints, 1, 1))  # (n_keypoints, 2, 2)
    Rs = np.tile(np.eye(2), (n_keypoints, 1, 1))  # (n_keypoints, 2, 2)

    # Compute covariance matrices
    cov_mats = []
    for i in range(n_keypoints):
        cov_mats.append([[1, 0], [0, 1]])
    cov_mats = np.array(cov_mats)

    return (
        jnp.array(m0s),
        jnp.array(S0s),
        jnp.array(As),
        jnp.array(cov_mats),
        jnp.array(Cs),
        jnp.array(Rs),
    )


def singlecam_optimize_smooth(
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
        return singlecam_smooth_min(
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

            def step(s, opt_state):
                loss, grads = jax.value_and_grad(loss_function)(
                    s, cov_mats_sub, y_subset, m0s_crop, S0s_crop, Cs_crop, As_crop, Rs_crop)
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

# ------------------------------------------------------------------------------------------------
# Routines that use the sequential kalman filter implementation to arrive at the NLL function
# Note: this code is set up to always run on CPU.
# ------------------------------------------------------------------------------------------------

def inner_smooth_min_routine(y, m0, S0, A, Q, C, R, ensemble_vars):
    # Run filtering with the current smooth_param
    _, _, nll = jax_forward_pass(y, m0, S0, A, Q, C, R, ensemble_vars)
    return nll


inner_smooth_min_routine_vmap = vmap(inner_smooth_min_routine, in_axes=(0, 0, 0, 0, 0, 0, 0))


def singlecam_smooth_min(smooth_param, cov_mats, ys, m0s, S0s, Cs, As, Rs, ensemble_vars):
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
