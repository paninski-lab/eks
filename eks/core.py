import jax
import numpy as np
import optax
from dynamax.nonlinear_gaussian_ssm import ParamsNLGSSM, extended_kalman_filter, \
    extended_kalman_smoother
from jax import numpy as jnp, vmap, jit, value_and_grad, lax
from typeguard import typechecked
from typing import Literal, Union, Optional, List, Tuple

from eks.marker_array import MarkerArray
from eks.utils import build_R_from_vars, crop_frames, crop_R


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


def params_nlgssm_for_keypoint(m0, S0, Q, s, R, f_fn, h_fn) -> ParamsNLGSSM:
    """
    Construct the ParamsNLGSSM for a single (keypoint) sequence.
    """
    return ParamsNLGSSM(
        initial_mean=jnp.asarray(m0),
        initial_covariance=jnp.asarray(S0),
        dynamics_function=f_fn,
        dynamics_covariance=jnp.asarray(s) * jnp.asarray(Q),
        emission_function=h_fn,
        emission_covariance=jnp.asarray(R),
    )


@typechecked
def optimize_smooth_param(
    Qs: jnp.ndarray,                 # (K, D, D)
    ys: np.ndarray,                  # (K, T, obs)
    m0s: jnp.ndarray,                # (K, D)
    S0s: jnp.ndarray,                # (K, D, D)
    Cs: jnp.ndarray,                 # (K, obs, D)
    As: jnp.ndarray,                 # (K, D, D)
    ensemble_vars: np.ndarray,       # (T, K, obs)
    s_frames: Optional[List] = None,
    smooth_param: Optional[Union[float, List[float]]] = None,
    blocks: Optional[List[List[int]]] = None,
    verbose: bool = False,
    # JIT-closed constants:
    lr: float = 0.25,
    s_bounds_log: tuple = (-8.0, 8.0),
    tol: float = 1e-3,
    safety_cap: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize the process-noise scale `s` (shared within each block of keypoints) by minimizing
    summed negative log-likelihood (NLL) under a *linear* state-space model using the
    Dynamax EKF filter (fast), then produce final trajectories via the EKF smoother.

    Model (per keypoint k):
        x_{t+1} = A_k x_t + w_t,   y_t = C_k x_t + v_t
        w_t ~ N(0, s * Q_k),       v_t ~ N(0, R_{k,t})

    where R_{k,t} is **time-varying**, built from ensemble variances:
        R_{k,t} = diag( clip( ensemble_vars[t, k, :], 1e-12, ∞ ) ).

    Args:
        Qs: (K, D, D) base process noise covariances Q_k per keypoint (scaled by `s`).
        ys: (K, T, obs) observations per keypoint across time.
        m0s: (K, D) initial state means per keypoint.
        S0s: (K, D, D) initial state covariances per keypoint.
        Cs: (K, obs, D) observation matrices C_k per keypoint.
        As: (K, D, D) transition matrices A_k per keypoint.
        ensemble_vars: (T, K, obs) per-frame ensemble variances for each keypoint’s obs dims;
                       used to construct time-varying R_{k,t}.
        s_frames: Optional list of frame indices used for NLL optimization (cropping the loss).
                  Final smoothing always runs on the full sequence.
        smooth_param: If provided, bypass optimization.
            • float/int: same `s` for all keypoints;
            • list[float] of length K: per-keypoint `s`.
        blocks: Optional list of lists of keypoint indices; each block shares a single `s`.
                Default: each keypoint forms its own block.
        verbose: If True, prints per-block optimization summaries.
        lr: Adam learning rate for optimizing log(s).
        s_bounds_log: (low, high) clamp for log(s) during optimization.
        tol: Relative tolerance on loss change for early stopping.
        safety_cap: Hard limit on iterations inside the jitted while-loop.

    Returns:
        s_finals: (K,) final `s` per keypoint (blockwise value broadcast to members).
        ms: (K, T, D) smoothed state means.
        Vs: (K, T, D, D) smoothed state covariances.

    Notes:
        • NLL is computed with EKF *filter*; outputs use EKF *smoother*.
        • Loss for a block is the sum of member keypoints’ NLLs (via vmap).
        • All jitted helpers close over optimizer/tol/bounds to avoid passing Python objects.
    """
    # -------------------- setup & time-varying R_t --------------------
    K, T, obs_dim = ys.shape
    if not blocks:
        blocks = [[k] for k in range(K)]
    if verbose:
        print(f"Correlated keypoint blocks: {blocks}")

    # Build time-varying R
    Rs = build_R_from_vars(np.swapaxes(ensemble_vars, 0, 1))
    Rs_j = jnp.asarray(Rs)

    # Device arrays once
    ys_j = jnp.asarray(ys)
    m0s_j = jnp.asarray(m0s)
    S0s_j = jnp.asarray(S0s)
    As_j = jnp.asarray(As)
    Qs_j = jnp.asarray(Qs)
    Cs_j = jnp.asarray(Cs)

    # Initial guesses per keypoint
    s_guess_per_k = np.empty(K, dtype=float)
    for k in range(K):
        g = float(compute_initial_guesses(ensemble_vars[:, k, :]) or 2.0)
        s_guess_per_k[k] = g if (np.isfinite(g) and g > 0.0) else 2.0

    # -------------------- choose or optimize s --------------------
    s_finals = np.empty(K, dtype=float)
    if smooth_param is not None:
        if isinstance(smooth_param, (int, float)):
            s_finals[:] = float(smooth_param)
        else:
            s_finals[:] = np.asarray(smooth_param, dtype=float)
    else:
        optimizer = optax.adam(float(lr))
        s_bounds_log_j = jnp.array(s_bounds_log, dtype=jnp.float32)
        tol_j = float(tol)

        def _params_linear(m0, S0, A, Q_base, s, R_any, C):
            f_fn = (lambda x, A=A: A @ x)         # linear dynamics
            h_fn = (lambda x, C=C: C @ x)         # linear emission
            return params_nlgssm_for_keypoint(m0, S0, Q_base, s, R_any, f_fn, h_fn)

        # NLL for a single keypoint with time-varying R_t
        def _nll_one_keypoint(log_s, y_k, m0_k, S0_k, A_k, Q_k, C_k, R_k_tv):
            s = jnp.exp(jnp.clip(log_s, s_bounds_log_j[0], s_bounds_log_j[1]))
            params = _params_linear(m0_k, S0_k, A_k, Q_k, s, R_k_tv, C_k)
            post = extended_kalman_filter(params, jnp.asarray(y_k))
            return -post.marginal_loglik

        # Sum NLL across all keypoints in the block
        def _nll_block(log_s, ys_b, m0s_b, S0s_b, As_b, Qs_b, Cs_b, Rs_b_tv):
            nlls = vmap(_nll_one_keypoint, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))(
                log_s, ys_b, m0s_b, S0s_b, As_b, Qs_b, Cs_b, Rs_b_tv
            )
            return jnp.sum(nlls)

        @jit
        def _opt_step(log_s, opt_state, ys_b, m0s_b, S0s_b, As_b, Qs_b, Cs_b, Rs_b_tv):
            loss, grad = value_and_grad(_nll_block)(
                log_s, ys_b, m0s_b, S0s_b, As_b, Qs_b, Cs_b, Rs_b_tv
            )
            updates, opt_state = optimizer.update(grad, opt_state)
            log_s = optax.apply_updates(log_s, updates)
            return log_s, opt_state, loss

        @jit
        def _run_tol_loop(log_s0, opt_state0, ys_b, m0s_b, S0s_b, As_b, Qs_b, Cs_b, Rs_b_tv):
            def cond(carry):
                _, _, prev_loss, iters, done = carry
                return jnp.logical_and(~done, iters < safety_cap)

            def body(carry):
                log_s, opt_state, prev_loss, iters, _ = carry
                log_s, opt_state, loss = _opt_step(
                    log_s, opt_state, ys_b, m0s_b, S0s_b, As_b, Qs_b, Cs_b, Rs_b_tv
                )
                rel_tol = tol_j * jnp.abs(jnp.log(jnp.maximum(prev_loss, 1e-12)))
                done = jnp.where(
                    jnp.isfinite(prev_loss),
                    jnp.linalg.norm(loss - prev_loss) < (rel_tol + 1e-6),
                    False
                )
                return (log_s, opt_state, loss, iters + 1, done)

            return lax.while_loop(
                cond, body, (log_s0, opt_state0, jnp.inf, jnp.array(0), jnp.array(False))
            )

        # Optimize per block (shared s within each block)
        for block in blocks:
            sel = jnp.asarray(block, dtype=int)

            # Crop frames for the loss (both y and R_t) if s_frames is provided
            if s_frames and len(s_frames) > 0:
                # Crop both y and R_t using the same frame spec -- each (T', obs)
                y_block_list = [crop_frames(ys[int(k)], s_frames) for k in block]
                R_block_list = [crop_R(Rs[int(k)], s_frames) for k in block]

                # Stack and jnp
                y_block = jnp.asarray(np.stack(y_block_list, axis=0))  # (B, T', obs)
                R_block = jnp.asarray(np.stack(R_block_list, axis=0))  # (B, T', obs, obs)
            else:
                y_block = ys_j[sel]  # (B, T, obs)
                R_block = Rs_j[sel]  # (B, T, obs, obs)

            m0_block = m0s_j[sel]
            S0_block = S0s_j[sel]
            A_block = As_j[sel]
            Q_block = Qs_j[sel]
            C_block = Cs_j[sel]

            s0 = float(np.mean([s_guess_per_k[k] for k in block]))
            log_s0 = jnp.array(np.log(max(s0, 1e-6)), dtype=jnp.float32)
            opt_state0 = optimizer.init(log_s0)

            log_s_f, opt_state_f, last_loss, iters_f, _done = _run_tol_loop(
                log_s0, opt_state0, y_block, m0_block, S0_block, A_block, Q_block, C_block,
                R_block
            )
            s_star = float(jnp.exp(jnp.clip(log_s_f, s_bounds_log_j[0], s_bounds_log_j[1])))
            for k in block:
                s_finals[k] = s_star
            if verbose:
                print(f"[Block {block}] s={s_star:.6g}, iters={int(iters_f)}, "
                      f"NLL={float(last_loss):.6f}")

    # -------------------- final smoother pass (full R_t) --------------------
    def _params_linear_for_k(k: int, s_val: float):
        A_k, C_k = As_j[k], Cs_j[k]
        f_fn = (lambda x, A=A_k: A @ x)
        h_fn = (lambda x, C=C_k: C @ x)
        return params_nlgssm_for_keypoint(
            m0s_j[k], S0s_j[k], Qs_j[k], s_val, Rs[k], f_fn, h_fn)

    means_list, covs_list = [], []
    for k in range(K):
        params_k = _params_linear_for_k(k, s_finals[k])
        sm = extended_kalman_smoother(params_k, ys_j[k])
        if hasattr(sm, "smoothed_means"):
            m_k, V_k = sm.smoothed_means, sm.smoothed_covariances
        else:
            m_k, V_k = sm.filtered_means, sm.filtered_covariances
        means_list.append(np.array(m_k))
        covs_list.append(np.array(V_k))

    ms = np.stack(means_list, axis=0)   # (K, T, D)
    Vs = np.stack(covs_list, axis=0)    # (K, T, D, D)
    return s_finals, ms, Vs
