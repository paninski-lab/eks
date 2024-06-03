import numpy as np
import pandas as pd
import jax
import jax.config
import jax.numpy as jnp
from eks.autotune_smooth_param import vectorized_compute_nll
from eks.utils import make_dlc_pandas_index
from eks.core import eks_zscore


def jax_ensemble_kalman_smoother_single_view(
        markers_3d_array, bodypart_list, smooth_param, s_frames, ensembling_mode='median',
        zscore_threshold=2, verbose=False):
    T = markers_3d_array.shape[1]
    n_keypoints = markers_3d_array.shape[2] // 3
    n_coords = 2

    # Compute ensemble statistics
    print("Calling jax_ensemble")
    ensemble_preds, ensemble_vars, keypoints_avg_dict = jax_ensemble(
        markers_3d_array, mode=ensembling_mode)

    # Calculate mean and adjusted observations
    mean_obs_dict, adjusted_obs_dict, scaled_ensemble_preds = jax_adjust_obs(
        keypoints_avg_dict, n_keypoints, ensemble_preds.copy())
    m0s, S0s, As, cov_mats, Cs, Rs, y_obs_array = initialize_kalman_filter(
        scaled_ensemble_preds, adjusted_obs_dict, n_keypoints)

    ms, Vs, nlls, nll_values = jax_singlecam_multicam_smooth_final(
        cov_mats, [smooth_param], y_obs_array, m0s, S0s, Cs, As, Rs, ensemble_vars)

    y_m_smooths = np.zeros((n_keypoints, T, n_coords))
    y_v_smooths = np.zeros((n_keypoints, T, n_coords, n_coords))
    eks_preds_array = np.zeros(y_m_smooths.shape)
    dfs = []
    df_dicts = []

    for k in range(n_keypoints):
        print(f"NLL is {nlls[k]} for {bodypart_list[k]}, smooth_param={smooth_param}")
        y_m_smooths[k] = np.dot(Cs[k], ms[k].T).T
        y_v_smooths[k] = np.swapaxes(np.dot(Cs[k], np.dot(Vs[k], Cs[k].T)), 0, 1)
        mean_x_obs = mean_obs_dict[3 * k]
        mean_y_obs = mean_obs_dict[3 * k + 1]
        # Computing zscore
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
    return df_dicts, smooth_param, nll_values


def jax_ensemble(markers_3d_array, mode='median'):
    """
    Computes ensemble median (or mean) and variance of a 3D array of DLC marker data using JAX.
    """
    markers_3d_array = jnp.array(markers_3d_array)  # Convert to JAX array
    n_frames = markers_3d_array.shape[1]
    n_keypoints = markers_3d_array.shape[2] // 3

    # Initialize output structures
    ensemble_preds = np.zeros((n_frames, n_keypoints, 2))
    ensemble_vars = np.zeros((n_frames, n_keypoints, 2))

    # Choose the appropriate JAX function based on the mode
    if mode == 'median':
        avg_func = lambda x: jnp.nanmedian(x, axis=0)
    elif mode == 'mean':
        avg_func = lambda x: jnp.nanmean(x, axis=0)
    elif mode == 'confidence_weighted_mean':
        avg_func = None
    else:
        raise ValueError(f"{mode} averaging not supported")

    def compute_stats(i):
        data_x = markers_3d_array[:, :, 3 * i]
        data_y = markers_3d_array[:, :, 3 * i + 1]
        data_likelihood = markers_3d_array[:, :, 3 * i + 2]

        if mode == 'confidence_weighted_mean':
            conf_per_keypoint = jnp.sum(data_likelihood, axis=0)
            mean_conf_per_keypoint = conf_per_keypoint / data_likelihood.shape[0]
            avg_x = jnp.sum(data_x * data_likelihood, axis=0) / conf_per_keypoint
            avg_y = jnp.sum(data_y * data_likelihood, axis=0) / conf_per_keypoint
            var_x = jnp.nanvar(data_x, axis=0) / mean_conf_per_keypoint
            var_y = jnp.nanvar(data_y, axis=0) / mean_conf_per_keypoint
        else:
            avg_x = avg_func(data_x)
            avg_y = avg_func(data_y)
            var_x = jnp.nanvar(data_x, axis=0)
            var_y = jnp.nanvar(data_y, axis=0)

        return avg_x, avg_y, var_x, var_y

    compute_stats_jit = jax.jit(compute_stats)
    stats = jax.vmap(compute_stats_jit)(jnp.arange(n_keypoints))

    avg_x, avg_y, var_x, var_y = stats

    keypoints_avg_dict = {}
    for i in range(n_keypoints):
        ensemble_preds[:, i, 0] = avg_x[i]
        ensemble_preds[:, i, 1] = avg_y[i]
        ensemble_vars[:, i, 0] = var_x[i]
        ensemble_vars[:, i, 1] = var_y[i]
        keypoints_avg_dict[2 * i] = avg_x[i]
        keypoints_avg_dict[2 * i + 1] = avg_y[i]

    # Convert outputs to JAX arrays
    ensemble_preds = jnp.array(ensemble_preds)
    ensemble_vars = jnp.array(ensemble_vars)
    keypoints_avg_dict = {k: jnp.array(v) for k, v in keypoints_avg_dict.items()}

    return ensemble_preds, ensemble_vars, keypoints_avg_dict


def jax_adjust_obs(keypoints_avg_dict, n_keypoints, scaled_ensemble_preds):
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
        cov_matrix = jnp.array(
            [[1, 0], [0, 1]])  # state covariance matrix; smaller = more smoothing
        C = jnp.array([[1, 0], [0, 1]])  # Measurement function
        R = jnp.eye(2)  # placeholder diagonal matrix for ensemble variance

        y_obs = scaled_ensemble_preds[:, i, :]

        return m0, S0, A, cov_matrix, C, R, y_obs

    # Use vmap to vectorize the initialization over all keypoints
    init_kalman_vmap = jax.vmap(init_kalman, in_axes=(0, 0, 0))
    m0s, S0s, As, cov_mats, Cs, Rs, y_obs_array = init_kalman_vmap(jnp.arange(n_keypoints),
                                                                   adjusted_x_obs_array,
                                                                   adjusted_y_obs_array)

    return m0s, S0s, As, cov_mats, Cs, Rs, y_obs_array


def jax_singlecam_multicam_smooth_final(cov_mats, s_finals, y, m0, S0, C, A, R, ensemble_vars):

    # Ensure s_finals is a JAX array and has the correct shape
    s_finals = jnp.array(s_finals)
    cov_mats = jnp.array(cov_mats)

    # Reshape s_finals to ensure it has at least one dimension
    if s_finals.ndim == 0:
        s_finals = s_finals[jnp.newaxis]

    Q = jax.vmap(lambda s: s * cov_mats)(s_finals)
    print("Calling jax_forward_pass...")
    mf, Vf, S, innovs, innov_covs = jax_forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    print("Calling jax_backward_pass...")
    ms, Vs, CV = jax_backward_pass(y, mf, Vf, S, A)
    print("Calling jax_compute_nll...")
    innovs = np.array(innovs)
    innov_covs = np.array(innov_covs)
    nll, nll_values = vectorized_compute_nll(innovs, innov_covs)
    return ms, Vs, nll, nll_values


def jax_forward_pass(y, m0s, S0s, Cs, Rs, As, Qs, ensemble_vars):
    n_keypoints, T, n_coords = y.shape
    ensemble_vars = jnp.array(ensemble_vars)

    def forward_step(i, y, m0s, S0s, Cs, Rs, As, Qs):
        # Index and initialize
        m0 = m0s[i]
        S0 = S0s[i]
        C = Cs[i]
        R = Rs[i]
        A = As[i]
        Q = Qs[i]
        ensemble_var = ensemble_vars[:, i, :]

        mf = jnp.zeros((T, m0.shape[0]))
        Vf = jnp.zeros((T, m0.shape[0], m0.shape[0]))
        S = jnp.zeros((T, m0.shape[0], m0.shape[0]))
        innovations = jnp.zeros((T, y.shape[2]))  # Assuming y is (n_keypoints, T, 2)
        innovation_cov = jnp.zeros((T, C.shape[0], C.shape[0]))

        for k in range(n_coords):
            R = R.at[k, k].set(ensemble_var[0][k])
        Ks0, _ = kalman_dot(y[i, 0, :] - jnp.dot(C, m0), S0, C, R)
        mf = mf.at[0].set(m0 + Ks0)
        Vf = Vf.at[0].set(S0 - Ks0)
        S = S.at[0].set(S0)
        innovations = innovations.at[0].set(y[i, 0] - jnp.dot(C, mf[0]))
        innovation_cov = innovation_cov.at[0].set(jnp.dot(C, jnp.dot(S0, C.T)) + R)

        # First update ------------
        # Propagate the state
        mf = mf.at[1].set(jnp.dot(A, mf[0]))
        S = S.at[0].set(jnp.dot(A, jnp.dot(Vf[0], A.T)) + Q[i])

        # Update R for time-varying observation variance
        R = R.at[0, 0].set(ensemble_var[0][0])
        R = R.at[1, 1].set(ensemble_var[0][1])

        # Update state estimate and covariance matrix
        innovations = innovations.at[1].set(y[i, 1, :] - jnp.dot(C, jnp.dot(A, mf[0, :])))
        Ks, _ = kalman_dot(innovations[1], S[0], C, R)
        mf = mf.at[1, :].add(Ks)

        Ks, innovation_cov_t = kalman_dot(jnp.dot(C, S[0]), S[0], C, R)
        Vf = Vf.at[1].set(S[0] - Ks)
        innovation_cov = innovation_cov.at[1].set(innovation_cov_t)

        # -------------------------

        def kalman_update(t, state):
            mf, Vf, S, innovations, innovation_cov, R, Ks = state

            # Propagate the state
            mf = mf.at[t].set(jnp.dot(A, mf[t - 1]))
            S = S.at[t - 1].set(jnp.dot(A, jnp.dot(Vf[t - 1], A.T)) + Q[i])

            # Update R for time-varying observation variance
            R = R.at[0, 0].set(ensemble_var[t][0])
            R = R.at[1, 1].set(ensemble_var[t][1])

            # Update state estimate and covariance matrix
            innovations = innovations.at[t].set(y[i, t, :] - jnp.dot(C, jnp.dot(A, mf[t - 1, :])))
            Ks, _ = kalman_dot(innovations[t], S[t - 1], C, R)
            mf = mf.at[t, :].add(Ks)

            Ks, innovation_cov_t = kalman_dot(jnp.dot(C, S[t - 1]), S[t - 1], C, R)
            Vf = Vf.at[t].set(S[t - 1] - Ks)
            innovation_cov = innovation_cov.at[t].set(innovation_cov_t)

            return mf, Vf, S, innovations, innovation_cov, R, Ks

        mf, Vf, S, innovations, innovation_cov, R, Ks = jax.lax.fori_loop(
            2, T, kalman_update, (mf, Vf, S, innovations, innovation_cov, R, Ks))
        return mf, Vf, S, innovations, innovation_cov

    forward_vmap = jax.vmap(forward_step, in_axes=(0, None, None, None, None, None, None, None))
    mfs, Vfs, Ss, innovations_array, innovation_covs_array = forward_vmap(
        jnp.arange(n_keypoints), y, m0s, S0s, Cs, Rs, As, Qs)

    return mfs, Vfs, Ss, innovations_array, innovation_covs_array


def kalman_dot(innovation, V, C, R):
    innovation_cov = R + jnp.dot(C, jnp.dot(V, C.T))
    innovation_cov_inv = jnp.linalg.solve(innovation_cov, innovation)
    Ks = jnp.dot(V, jnp.dot(C.T, innovation_cov_inv))
    return Ks, innovation_cov


def jax_backward_pass(ys, mfs, Vfs, Ss, As):
    n_keypoints, T, n_coords = ys.shape

    def backward_step(k, y_k, mf_k, Vf_k, S_k, A_k):
        ms = jnp.zeros((T, mf_k.shape[1]))
        Vs = jnp.zeros((T, mf_k.shape[1], mf_k.shape[1]))
        CV = jnp.zeros((T - 1, mf_k.shape[1], mf_k.shape[1]))

        ms = ms.at[-1].set(mf_k[-1])
        Vs = Vs.at[-1].set(Vf_k[-1])

        def smoothing_update(i, state):
            ms, Vs, CV = state
            i = T - 2 - i  # Workaround for lack of jax.lax.fori_loop custom incrementing

            J = jax.scipy.linalg.solve(S_k[i], jnp.dot(A_k, Vf_k[i]), assume_a='pos').T
            Vs = Vs.at[i].set(Vf_k[i] + jnp.dot(J, jnp.dot(Vs[i + 1] - S_k[i], J.T)))
            ms = ms.at[i].set(mf_k[i] + jnp.dot(J, ms[i + 1] - jnp.dot(A_k, mf_k[i])))
            CV = CV.at[i].set(jnp.dot(Vs[i + 1], J.T))
            return ms, Vs, CV

        ms, Vs, CV = jax.lax.fori_loop(0, T - 1, smoothing_update, (ms, Vs, CV))

        return ms, Vs, CV

    backward_vmap = jax.vmap(backward_step, in_axes=(0, 0, 0, 0, 0, 0))
    ms, Vs, CV = backward_vmap(jnp.arange(n_keypoints), ys, mfs, Vfs, Ss, As)

    return ms, Vs, CV
