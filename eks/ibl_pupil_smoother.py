import os
import warnings

from dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_filter,
    extended_kalman_smoother,
)

import jax
import numpy as np
import optax
import pandas as pd
from jax import jit, lax, value_and_grad
from jax import numpy as jnp
from numbers import Real
from typeguard import typechecked
from typing import List, Optional, Sequence, Tuple

from eks.core import ensemble, params_nlgssm_for_keypoint
from eks.marker_array import MarkerArray, input_dfs_to_markerArray
from eks.utils import build_R_from_vars, crop_frames, format_data, make_dlc_pandas_index


@typechecked
def get_pupil_location(dlc: dict) -> np.ndarray:
    """get mean of both pupil diameters
    d1 = top - bottom, d2 = left - right
    and in addition assume it's a circle and
    estimate diameter from other pairs of points
    Author: Michael Schartner
    """
    s = 1
    t = np.vstack((dlc['pupil_top_r_x'], dlc['pupil_top_r_y'])).T / s
    b = np.vstack((dlc['pupil_bottom_r_x'], dlc['pupil_bottom_r_y'])).T / s
    le = np.vstack((dlc['pupil_left_r_x'], dlc['pupil_left_r_y'])).T / s
    r = np.vstack((dlc['pupil_right_r_x'], dlc['pupil_right_r_y'])).T / s
    center = np.zeros(t.shape)

    # ok if either top or bottom is nan in x-dir
    tmp_x1 = np.nanmedian(np.hstack([t[:, 0, None], b[:, 0, None]]), axis=1)
    # both left and right must be present in x-dir
    tmp_x2 = np.median(np.hstack([r[:, 0, None], le[:, 0, None]]), axis=1)
    center[:, 0] = np.nanmedian(np.hstack([tmp_x1[:, None], tmp_x2[:, None]]), axis=1)

    # both top and bottom must be present in ys-dir
    tmp_y1 = np.median(np.hstack([t[:, 1, None], b[:, 1, None]]), axis=1)
    # ok if either left or right is nan in ys-dir
    tmp_y2 = np.nanmedian(np.hstack([r[:, 1, None], le[:, 1, None]]), axis=1)
    center[:, 1] = np.nanmedian(np.hstack([tmp_y1[:, None], tmp_y2[:, None]]), axis=1)
    return center


@typechecked
def get_pupil_diameter(dlc: dict) -> np.ndarray:
    """
    from: https://int-brain-lab.github.io/iblenv/_modules/brainbox/behavior/dlc.html
    Estimates pupil diameter by taking median of different computations.

    The two most straightforward estimates: d1 = top - bottom, d2 = left - right
    In addition, assume the pupil is a circle and estimate diameter from other pairs of points

    :param dlc: dlc pqt table with pupil estimates, should be likelihood thresholded (e.g. at 0.9)
    :return: np.array, pupil diameter estimate for each time point, shape (n_frames,)
    """
    diameters = []
    # Get the x,ys coordinates of the four pupil points
    top, bottom, left, right = [
        np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
        for point in ['top', 'bottom', 'left', 'right']
    ]
    # First compute direct diameters
    diameters.append(np.linalg.norm(top - bottom, axis=0))
    diameters.append(np.linalg.norm(left - right, axis=0))

    # For non-crossing edges, estimate diameter via circle assumption
    for pair in [(top, left), (top, right), (bottom, left), (bottom, right)]:
        diameters.append(np.linalg.norm(pair[0] - pair[1], axis=0) * 2 ** 0.5)

    # Ignore all nan runtime warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(diameters, axis=0)


def add_mean_to_array(pred_arr, keys, mean_x, mean_y):
    pred_arr_copy = pred_arr.copy()
    processed_arr_dict = {}
    for i, key in enumerate(keys):
        if 'x' in key:
            processed_arr_dict[key] = pred_arr_copy[:, i] + mean_x
        else:
            processed_arr_dict[key] = pred_arr_copy[:, i] + mean_y
    return processed_arr_dict


@typechecked
def fit_eks_pupil(
    input_source: str | list,
    save_file: str,
    smooth_params: list | None = None,
    s_frames: list | None = None,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
) -> tuple:
    """Fit the Ensemble Kalman Smoother for the ibl-pupil dataset.

    Args:
        input_source: directory path or list of CSV file paths. If a directory path, all files
            within this directory will be used.
        save_file: File to save output dataframe.
        smooth_params: [diameter param, center of mass param]
            each value should be in (0, 1); closer to 1 means more smoothing
        s_frames: Frames for automatic optimization if needed.
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        verbose: Extra print statements if True

    Returns:
        tuple:
            df_smoothed (pd.DataFrame)
            smooth_params (list): Final smoothing parameters used.
            input_dfs_list (list): List of input DataFrames.
            keypoint_names (list): List of keypoint names.

    """
    # pupil smoother only works for a pre-specified set of points
    # NOTE: this order MUST be kept
    bodypart_list = ['pupil_top_r', 'pupil_bottom_r', 'pupil_right_r', 'pupil_left_r']

    # Load and format input files
    input_dfs_list, _ = format_data(input_source)
    print(f"Input data loaded for keypoints: {bodypart_list}")
    marker_array = input_dfs_to_markerArray([input_dfs_list], bodypart_list, [""])

    # Run the ensemble Kalman smoother
    df_smoothed, smooth_params_final = ensemble_kalman_smoother_ibl_pupil(
        marker_array=marker_array,
        keypoint_names=bodypart_list,
        smooth_params=smooth_params,
        s_frames=s_frames,
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
def ensemble_kalman_smoother_ibl_pupil(
    marker_array: MarkerArray,
    keypoint_names: list,
    smooth_params: list | None = None,
    s_frames: list | None = None,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False
) -> tuple:
    """Perform Ensemble Kalman Smoothing on pupil data.

    Args:
        marker_array: MarkerArray object containing marker data.
            Shape (n_models, n_cameras, n_frames, n_keypoints, 3 (for x, y, likelihood))
        markers_list: pd.DataFrames
            each list element is a dataframe of predictions from one ensemble member
        keypoint_names: List of body parts to run smoothing on
        smooth_params: contains smoothing parameters for diameter and center of mass
        s_frames: frames for automatic optimization if s is not provided
        avg_mode: mode for averaging across ensemble
            'median' | 'mean'
        var_mode: mode for computing ensemble variance
            'var' | 'confidence_weighted_var'
        verbose: True to print out details

    Returns:
        tuple:
            smoothed markers dataframe
            final smooth params values
            final nll

    """
    n_models, n_cameras, n_frames, n_keypoints, n_data_fields = marker_array.shape
    keys = [f'{kp}_{coord}' for kp in keypoint_names for coord in ['x', 'y']]

    # Compute ensemble information
    # MarkerArray (1, 1, n_frames, n_keypoints, 5 (x, y, var_x, var_y, likelihood))
    ensemble_marker_array = ensemble(marker_array, avg_mode=avg_mode, var_mode=var_mode)
    emA_unsmoothed_preds = ensemble_marker_array.slice_fields("x", "y")
    emA_vars = ensemble_marker_array.slice_fields("var_x", "var_y")
    emA_likes = ensemble_marker_array.slice_fields("likelihood")

    # Extract stacked arrays (predicted coordinates, variances, likelihoods)
    ensemble_preds = emA_unsmoothed_preds.get_array(squeeze=True).reshape(n_frames, -1)
    ensemble_vars = emA_vars.get_array(squeeze=True).reshape(n_frames, -1)
    ensemble_likes = emA_likes.get_array(squeeze=True)

    # Compute center of mass + diameter
    pupil_diameters = get_pupil_diameter({key: ensemble_preds[:, i] for i, key in enumerate(keys)})
    pupil_locations = get_pupil_location({key: ensemble_preds[:, i] for i, key in enumerate(keys)})
    mean_x_obs = np.mean(pupil_locations[:, 0])
    mean_y_obs = np.mean(pupil_locations[:, 1])

    # Center predictions
    x_t_obs, y_t_obs = pupil_locations[:, 0] - mean_x_obs, pupil_locations[:, 1] - mean_y_obs

    # -------------------------------------------------------
    # Set values for Kalman filter (Specific to this dataset)
    # -------------------------------------------------------
    # initial state: mean
    m0 = np.asarray([np.mean(pupil_diameters), 0.0, 0.0])

    # diagonal: var
    S0 = np.asarray([
        [np.nanvar(pupil_diameters), 0.0, 0.0],
        [0.0, np.nanvar(x_t_obs), 0.0],
        [0.0, 0.0, np.nanvar(y_t_obs)]
    ])

    # Measurement function
    C = np.asarray([
        [0, 1, 0], [-.5, 0, 1],
        [0, 1, 0], [.5, 0, 1],
        [.5, 1, 0], [0, 0, 1],
        [-.5, 1, 0], [0, 0, 1]
    ])

    centered_ensemble_preds = ensemble_preds.copy()
    # subtract COM means from the ensemble predictions
    for i in range(ensemble_preds.shape[1]):
        if i % 2 == 0:
            centered_ensemble_preds[:, i] -= mean_x_obs
        else:
            centered_ensemble_preds[:, i] -= mean_y_obs
    y_obs = centered_ensemble_preds

    # -------------------------------------------------------
    # Perform filtering with SINGLE PAIR of diameter_s, com_s
    # -------------------------------------------------------
    s_finals, ms, Vs, nll = pupil_optimize_smooth(
        y_obs, m0, S0, C, ensemble_vars,
        np.var(pupil_diameters), np.var(x_t_obs), np.var(y_t_obs), s_frames, smooth_params,
        verbose=verbose)
    if verbose:
        print(f"diameter_s={s_finals[0]}, com_s={s_finals[1]}")
    # Smoothed posterior over ys
    y_m_smooth = np.dot(C, ms.T).T
    y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

    # --------------------------------------
    # cleanup
    # --------------------------------------

    # collect data
    processed_arr_dict = add_mean_to_array(y_m_smooth, keys, mean_x_obs, mean_y_obs)
    key_pair_list = [
        ['pupil_top_r_x', 'pupil_top_r_y'],
        ['pupil_right_r_x', 'pupil_right_r_y'],
        ['pupil_bottom_r_x', 'pupil_bottom_r_y'],
        ['pupil_left_r_x', 'pupil_left_r_y'],
    ]
    ensemble_indices = [(0, 1), (4, 5), (2, 3), (6, 7)]
    data_arr = []
    for i, key_pair in enumerate(key_pair_list):
        # keep track of labels for each data entry
        labels = []
        # smoothed x vals
        data_arr.append(processed_arr_dict[key_pair[0]])
        labels.append('x')
        # smoothed y vals
        data_arr.append(processed_arr_dict[key_pair[1]])
        labels.append('y')
        # mean likelihood
        data_arr.append(ensemble_likes[:, i])
        labels.append('likelihood')
        # x vals ensemble median
        data_arr.append(ensemble_preds[:, ensemble_indices[i][0]])
        labels.append('x_ens_median')
        # y vals ensemble median
        data_arr.append(ensemble_preds[:, ensemble_indices[i][1]])
        labels.append('y_ens_median')
        # x vals ensemble variance
        data_arr.append(ensemble_vars[:, ensemble_indices[i][0]])
        labels.append('x_ens_var')
        # y vals ensemble variance
        data_arr.append(ensemble_vars[:, ensemble_indices[i][1]])
        labels.append('y_ens_var')
        # x vals posterior variance
        data_arr.append(y_v_smooth[:, i, i])
        labels.append('x_posterior_var')
        # y vals posterior variance
        data_arr.append(y_v_smooth[:, i + 1, i + 1])
        labels.append('y_posterior_var')

    data_arr = np.asarray(data_arr)

    # put data in dataframe
    pdindex = make_dlc_pandas_index(keypoint_names, labels=labels)
    markers_df = pd.DataFrame(data_arr.T, columns=pdindex)

    return markers_df, s_finals


@typechecked
def pupil_optimize_smooth(
    ys: np.ndarray,                 # (T, 8) centered obs
    m0: np.ndarray,                 # (3,)
    S0: np.ndarray,                 # (3,3)
    C: np.ndarray,                  # (8,3)
    ensemble_vars: np.ndarray,      # (T, 8)
    diameters_var: Real,
    x_var: float,
    y_var: float,
    s_frames: Optional[List[Tuple[Optional[int], Optional[int]]]] = [(1, 2000)],
    smooth_params: list | None = None,   # [diam_s, com_s] in (0,1)
    maxiter: int = 1000,            # retained (unused with tol-loop)
    verbose: bool = False,
    # optimizer/loop knobs
    lr: float = 5e-3,
    tol: float = 1e-6,
    safety_cap: int = 5000,
) -> tuple:
    """
    Dynamax backend: optimize [s_diameter, s_com] with EKF NLL, then EKF smoother.

    Returns:
        s_finals (list[float]), ms (T,3), Vs (T,3,3), nll (float)
    """

    # logistic reparam to keep s in (eps,1-eps)
    def _to_stable_s(u, eps=1e-3):
        return jax.nn.sigmoid(u) * (1.0 - 2 * eps) + eps

    # crop ys and ev for the loss if s_frames provided
    if s_frames and len(s_frames) > 0:
        y_cropped = crop_frames(ys, s_frames)              # (T', 8)
        ev_cropped = crop_frames(ensemble_vars, s_frames)  # (T', 8)
    else:
        y_cropped, ev_cropped = ys, ensemble_vars

    # build time-varying R_t for loss and for final smoothing (full sequence)
    R_loss = build_R_from_vars(ev_cropped)     # (T' ,8,8)
    R_full = build_R_from_vars(ensemble_vars)  # (T  ,8,8)

    # jnp once
    y_c = jnp.asarray(y_cropped)
    R_loss = jnp.asarray(R_loss)
    m0_j, S0_j, C_j = jnp.asarray(m0), jnp.asarray(S0), jnp.asarray(C)
    y_full = jnp.asarray(ys)
    R_full = jnp.asarray(R_full)

    # local params builder using your NLGSSM wrapper; pass Q_exact with s=1.0
    def _params_linear(m0, S0, A, Q_exact, R_any, C):
        f_fn = (lambda x, A=A: A @ x)
        h_fn = (lambda x, C=C: C @ x)
        return params_nlgssm_for_keypoint(m0, S0, Q_exact, 1.0, R_any, f_fn, h_fn)

    # EKF NLL for a given unconstrained u = [u_d, u_c]
    def _nll_from_u(u: jnp.ndarray) -> jnp.ndarray:
        s_d, s_c = _to_stable_s(u)
        A = jnp.diag(jnp.array([s_d, s_c, s_c]))
        Q = jnp.diag(jnp.array([
            diameters_var * (1.0 - s_d**2),
            x_var * (1.0 - s_c**2),
            y_var * (1.0 - s_c**2),
        ]))
        params = _params_linear(m0_j, S0_j, A, Q, R_loss, C_j)
        post = extended_kalman_filter(params, y_c)
        return -post.marginal_loglik

    optimizer = optax.adam(lr)
    if smooth_params is None or smooth_params[0] is None or smooth_params[1] is None:
        # init near your old guess (invert logistic)
        s0 = jnp.array([0.99, 0.98])
        u0 = jnp.log(s0 / (1.0 - s0))
        opt_state0 = optimizer.init(u0)

        @jit
        def _opt_step(u, opt_state):
            loss, grad = value_and_grad(_nll_from_u)(u)
            updates, opt_state = optimizer.update(grad, opt_state)
            u = optax.apply_updates(u, updates)
            return u, opt_state, loss

        @jit
        def _run_tol_loop(u0, opt_state0):
            def cond(carry):
                _, _, prev_loss, iters, done = carry
                return jnp.logical_and(~done, iters < safety_cap)

            def body(carry):
                u, opt_state, prev_loss, iters, _ = carry
                u, opt_state, loss = _opt_step(u, opt_state)
                rel_tol = tol * jnp.abs(jnp.log(jnp.maximum(prev_loss, 1e-12)))
                done = jnp.where(jnp.isfinite(prev_loss),
                                 jnp.linalg.norm(loss - prev_loss) < (rel_tol + 1e-6),
                                 False)
                return (u, opt_state, loss, iters + 1, done)
            return lax.while_loop(
                cond, body, (u0, opt_state0, jnp.inf, jnp.array(0), jnp.array(False))
            )

        u_f, opt_state_f, last_loss, iters_f, _ = _run_tol_loop(u0, opt_state0)
        s_opt = _to_stable_s(u_f)
        if verbose:
            print(f"[pupil/dynamax] iters={int(iters_f)}  s_diam={float(s_opt[0]):.6f}  "
                  f"s_com={float(s_opt[1]):.6f}  NLL={float(last_loss):.6f}")
    else:
        s_user = jnp.clip(jnp.asarray(smooth_params, dtype=jnp.float32), 1e-3, 1 - 1e-3)
        s_opt = s_user

    # final smoother on full sequence with full R_t
    s_d, s_c = float(s_opt[0]), float(s_opt[1])
    ms, Vs, nll = pupil_smooth(
        smooth_params=[s_d, s_c],
        ys=y_full, m0=m0_j, S0=S0_j, C=C_j, R=R_full,
        diameters_var=diameters_var, x_var=x_var, y_var=y_var,
        return_full=True
    )
    return [s_d, s_c], np.asarray(ms), np.asarray(Vs), float(nll)


@typechecked
def pupil_smooth(
    smooth_params: Sequence[float],      # [s_diam, s_com] in (0,1)
    ys: np.ndarray | jnp.ndarray,        # (T, 8)
    m0: np.ndarray | jnp.ndarray,        # (3,)
    S0: np.ndarray | jnp.ndarray,        # (3,3)
    C: np.ndarray | jnp.ndarray,         # (8,3)
    R: np.ndarray | jnp.ndarray,         # (T, 8, 8) time-varying obs covariance
    diameters_var: Real,
    x_var: float,
    y_var: float,
    return_full: bool = False,
):
    """
    One EKF forward (and optional smoother) using Dynamax NLGSSM with:
      A = diag([s_d, s_c, s_c]) and Q = diag([σ_d^2(1-s_d^2), σ_x^2(1-s_c^2), σ_y^2(1-s_c^2)]).
      R_t = diag(ensemble_vars[t]) (or provided via _R_override).
    """
    ys = jnp.asarray(ys)
    m0 = jnp.asarray(m0)
    S0 = jnp.asarray(S0)
    C = jnp.asarray(C)

    s_d = jnp.clip(jnp.asarray(smooth_params[0]), 1e-3, 1 - 1e-3)
    s_c = jnp.clip(jnp.asarray(smooth_params[1]), 1e-3, 1 - 1e-3)

    A = jnp.diag(jnp.array([s_d, s_c, s_c]))
    Q = jnp.diag(jnp.array([
        diameters_var * (1.0 - s_d**2),
        x_var * (1.0 - s_c**2),
        y_var * (1.0 - s_c**2),
    ]))

    # linear f/h closures
    f_fn = (lambda x, A=A: A @ x)
    h_fn = (lambda x, C=C: C @ x)

    # build NLGSSM params; pass Q as exact and s=1.0 to avoid extra scaling
    params = params_nlgssm_for_keypoint(m0, S0, Q, 1.0, R, f_fn, h_fn)

    filt = extended_kalman_filter(params, ys)
    nll = -filt.marginal_loglik
    if not return_full:
        return nll

    sm = extended_kalman_smoother(params, ys)
    if hasattr(sm, "smoothed_means"):
        ms = sm.smoothed_means
        Vs = sm.smoothed_covariances
    else:
        ms = sm.filtered_means
        Vs = sm.filtered_covariances
    return ms, Vs, -filt.marginal_loglik
