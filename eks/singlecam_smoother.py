import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from eks.utils import make_dlc_pandas_index, crop_frames
from eks.core import eks_zscore, jax_ensemble, jax_forward_pass, jax_backward_pass, \
    compute_covariance_matrix, jax_compute_nll, compute_initial_guesses
from scipy.optimize import minimize


def ensemble_kalman_smoother_singlecam(
        markers_3d_array, bodypart_list, smooth_param, s_frames, ensembling_mode='median',
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
    tuple: Dataframes with smoothed predictions, final smoothing parameters, negative log-likelihood values.
    """

    # Detect GPU
    try:
        _ = jax.device_put(jax.numpy.ones(1), device=jax.devices('gpu')[0])
        print("Using GPU")
    except:
        print("Using CPU")

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
    s_finals, ms, Vs, nlls, nll_values = singlecam_optimize_smooth(
        cov_mats, ys, m0s, S0s, Cs, As, Rs, ensemble_vars,
        s_frames, smooth_param
    )

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

    return df_dicts, s_finals, nll_values


def adjust_observations(keypoints_avg_dict, n_keypoints, scaled_ensemble_preds):
    """
    Adjust observations by computing mean and adjusted observations for each keypoint.

    Parameters:
    keypoints_avg_dict (dict): Dictionary of keypoints averages.
    n_keypoints (int): Number of keypoints.
    scaled_ensemble_preds (np.ndarray): Scaled ensemble predictions.

    Returns:
    tuple: Mean observations dictionary, adjusted observations dictionary, scaled ensemble predictions.
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
        s_frames, smooth_param, blocks=[]):
    """
    Optimize smoothing parameter and perform smoothing.

    Parameters:
    cov_mats (np.ndarray): Covariance matrices.
    ys (np.ndarray): Observations.
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
    tuple: Final smoothing parameters, smoothed means, smoothed covariances, negative log-likelihoods, negative log-likelihood values.
    """

    n_keypoints = ys.shape[0]
    s_finals = []
    if blocks == []:
        for n in range(n_keypoints):
            blocks.append([n])
    # Optimize smooth_param
    if smooth_param is None:
        guesses = []
        cropped_ys = []
        for k in range(n_keypoints):
            guesses.append(compute_initial_guesses(ensemble_vars[:, k, :]))
            # Unpack s_frames
            cropped_ys.append(crop_frames(ys[k], s_frames))

        # Minimize negative log likelihood serially for each keypoint
        for block in blocks:
            s_final = minimize(
                singlecam_smooth_min,
                x0=guesses[k],  # initial smooth param guess
                args=(block,
                      cov_mats,
                      cropped_ys,
                      m0s,
                      S0s,
                      Cs,
                      As,
                      Rs),
                method='Nelder-Mead',
                bounds=[(0, None)],
            )
            for b in block:
                s = s_final.x[0]
                print(f's={s} for keypoint {b}')
                s_finals.append(s_final.x[0])
    else:
        s_finals = [smooth_param]

    # Final smooth with optimized s
    ms, Vs, nll, nll_values = singlecam_smooth_final(
        cov_mats, s_finals,
        ys, m0s, S0s, Cs, As, Rs)

    return s_finals, ms, Vs, nll, nll_values


def singlecam_smooth_min(
        smooth_param, block, cov_mats, ys, m0s, S0s, Cs, As, Rs):
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
    nll_sum = 0
    for b in block:
        Q = Qs[b]
        y = ys[b]
        m0 = m0s[b]
        S0 = S0s[b]
        C = Cs[b]
        A = As[b]
        R = Rs[b]
        # Run filtering with the current smooth_param
        _, _, _, innovs, innov_cov = jax_forward_pass(y, m0, S0, A, Q, C, R)
        # Compute the negative log-likelihood based on innovations and their covariance
        nll, nll_values = jax_compute_nll(innovs, innov_cov)
        nll_sum += nll
    return nll_sum


def singlecam_smooth_final(cov_mats, s_finals, ys, m0s, S0s, Cs, As, Rs):
    """
    Perform final smoothing with the optimized smoothing parameters.

    Parameters:
    cov_mats (np.ndarray): Covariance matrices.
    s_finals (np.ndarray): Final smoothing parameters.
    ys (np.ndarray): Observations.
    m0s (np.ndarray): Initial mean state.
    S0s (np.ndarray): Initial state covariance.
    Cs (np.ndarray): Measurement function.
    As (np.ndarray): State-transition matrix.
    Rs (np.ndarray): Measurement noise covariance.

    Returns:
    tuple: Smoothed means, smoothed covariances, negative log-likelihoods, negative log-likelihood values.
    """

    # Initialize
    s_finals = jnp.array(s_finals)
    if s_finals.ndim == 0:
        s_finals = s_finals[jnp.newaxis]
    cov_mats = jnp.array(cov_mats)
    n_keypoints = ys.shape[0]
    ms_array = []
    Vs_array = []
    nlls = []
    nll_values_array = []
    Qs = jax.vmap(lambda s: s * cov_mats)(s_finals)[0]

    # Run forward and backward pass for each keypoint
    for k in range(n_keypoints):
        mf, Vf, S, innovs, innov_covs = jax_forward_pass(
            ys[k], m0s[k], S0s[k], As[k], Qs[k], Cs[k], Rs[k])
        ms, Vs = jax_backward_pass(mf, Vf, S, As[k])
        ms_array.append(ms)
        Vs_array.append(Vs)
        nll, nll_values = jax_compute_nll(innovs, innov_covs)
        nlls.append(nll)
        nll_values_array.append(nll_values)

    return ms_array, Vs_array, nlls, nll_values_array