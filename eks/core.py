import numpy as np

def ensemble(markers_list, keys, mode='median'):
    """Computes ensemble median (or mean) and variance of list of DLC marker dataframes.

    Args:
        markers_list: list
            List of DLC marker dataframes.
        keys: list
            List of keys in each marker dataframe.
        mode: string
            Averaging mode which includes 'median', 'mean', or 'confidence_weighted_mean'.

    Returns:
        ensemble_preds: np.ndarray
            Shape (samples, n_keypoints).
        ensemble_vars: np.ndarray
            Shape (samples, n_keypoints).
        keypoints_avg_dict: dict
            Keys: marker keypoints, values: shape (samples).
    """
    if mode not in {'median', 'mean', 'confidence_weighted_mean'}:
        raise ValueError(f"{mode} averaging not supported")

    if mode == 'median':
        average_func = np.nanmedian
    elif mode == 'mean':
        average_func = np.nanmean

    ensemble_preds = []
    ensemble_vars = []
    keypoints_avg_dict = {}
    for key in keys:
        stack = np.array([df[key] for df in markers_list]).T

        if mode != 'confidence_weighted_mean':
            avg = average_func(stack, axis=1)
            var = np.nanvar(stack, axis=1)
        else:
            likelihood_key = key[:-1] + 'likelihood'
            if likelihood_key not in markers_list[0]:
                raise ValueError(f"{likelihood_key} needs to be in your marker_df to use {mode}")
            likelihood_stack = np.array([df[likelihood_key] for df in markers_list]).T
            conf_per_keypoint = np.sum(likelihood_stack, axis=1)
            mean_conf_per_keypoint = conf_per_keypoint / likelihood_stack.shape[1]
            avg = np.sum(stack * likelihood_stack, axis=1) / conf_per_keypoint
            var = np.nanvar(stack, axis=1) / mean_conf_per_keypoint

        ensemble_preds.append(avg)
        ensemble_vars.append(var)
        keypoints_avg_dict[key] = avg

    ensemble_preds = np.asarray(ensemble_preds).T
    ensemble_vars = np.asarray(ensemble_vars).T
    return ensemble_preds, ensemble_vars, keypoints_avg_dict


def vectorized_ensemble(markers_3d_array, keys, mode='median'):
    """Computes ensemble median (or mean) and variance of a 3D array of DLC marker data."""
    n_models, frames, _ = markers_3d_array.shape
    n_keypoints = len(keys) // 2
    # Initialize output structures
    ensemble_preds = np.zeros((frames, n_keypoints, 2))
    ensemble_vars = np.zeros((frames, n_keypoints, 2))
    keypoints_avg_dict = {}

    # Choose the appropriate numpy function based on the mode
    if mode == 'median':
        avg_func = np.nanmedian
    elif mode == 'mean':
        avg_func = np.nanmean
    elif mode == 'confidence_weighted_mean':
        avg_func = None
    else:
        raise ValueError(f"{mode} averaging not supported")

    # Process each pair of keys (x, y) to compute statistics
    for i in range(n_keypoints):
        data_x = markers_3d_array[:, :, 3 * i]
        data_y = markers_3d_array[:, :, 3 * i + 1]
        data_likelihood = markers_3d_array[:, :, 3 * i + 2]

        if mode == 'confidence_weighted_mean':
            conf_per_keypoint = np.sum(data_likelihood, axis=0)
            mean_conf_per_keypoint = conf_per_keypoint / data_likelihood.shape[0]
            avg_x = np.sum(data_x * data_likelihood, axis=0) / conf_per_keypoint
            avg_y = np.sum(data_y * data_likelihood, axis=0) / conf_per_keypoint
            var_x = np.nanvar(data_x, axis=0) / mean_conf_per_keypoint
            var_y = np.nanvar(data_y, axis=0) / mean_conf_per_keypoint
        else:
            avg_x = avg_func(data_x, axis=0)
            avg_y = avg_func(data_y, axis=0)
            var_x = np.nanvar(data_x, axis=0)
            var_y = np.nanvar(data_y, axis=0)

        ensemble_preds[:, i, 0] = avg_x
        ensemble_preds[:, i, 1] = avg_y
        ensemble_vars[:, i, 0] = var_x
        ensemble_vars[:, i, 1] = var_y

        keypoints_avg_dict[keys[2 * i]] = avg_x
        keypoints_avg_dict[keys[2 * i + 1]] = avg_y

    return ensemble_preds, ensemble_vars, keypoints_avg_dict



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
        mf: np.ndarray
            shape (samples, n_keypoints)
        Vf: np.ndarray
            shape (samples, n_latents, n_latents)
        S: np.ndarray
            shape (samples, n_latents, n_latents)
        innovations: np.ndarray
            shape (samples, n_keypoints)
        innovation_cov: np.ndarray
            shape (samples, n_keypoints, n_keypoints)
    """
    T = y.shape[0]
    mf = np.zeros(shape=(T, m0.shape[0]))
    Vf = np.zeros(shape=(T, m0.shape[0], m0.shape[0]))
    S = np.zeros(shape=(T, m0.shape[0], m0.shape[0]))
    innovations = np.zeros((T, y.shape[1]))
    innovation_cov = np.zeros((T, C.shape[0], C.shape[0]))
    # time-varying observation variance
    for i in range(ensemble_vars.shape[1]):
        R[i, i] = ensemble_vars[0][i]
    K_array, _ = kalman_dot(y[0, :] - np.dot(C, m0), S0, C, R)
    mf[0] = m0 + K_array
    Vf[0, :] = S0 - K_array
    S[0] = S0
    innovations[0] = y[0] - np.dot(C, mf[0])
    innovation_cov[0] = np.dot(C, np.dot(S0, C.T)) + R

    # Kalman filter update for subsequent time steps
    for t in range(1, T):
        # Propagate the state
        mf[t, :] = np.dot(A, mf[t - 1, :])
        S[t - 1] = np.dot(A, np.dot(Vf[t - 1, :], A.T)) + Q

        if np.sum(~np.isnan(y[t, :])) >= 2:  # Check if any value in y[t] is not NaN
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


def vectorized_forward_pass(y, m0_array, S0_array, C_array, R_array, A_array, Q_array,
                            ensemble_vars):
    n_keypoints, T, n_coords = y.shape[0], y.shape[1], y.shape[2]
    # Initialize arrays for each keypoint
    mf_array = np.zeros((n_keypoints, T, n_coords))
    Vf_array = np.zeros((n_keypoints, T, n_coords, n_coords))
    S_array = np.zeros((n_keypoints, T, n_coords, n_coords))
    innovations_array = np.zeros((n_keypoints, T, n_coords))  # Assuming y is (n_keypoints, T, 2)
    innovation_cov_array = np.zeros((n_keypoints, T, C_array.shape[1], C_array.shape[1]))

    for i in range(n_keypoints - 1):
        y_obs = y[i]
        m0 = m0_array[i]
        S0 = S0_array[i]
        C = C_array[i]
        R = R_array[i]
        A = A_array[i]
        if i < len(Q_array):
            Q = Q_array[i]
        ensemble_var = ensemble_vars[:, i, :]

        # Initialize mf, Vf, and S for the current keypoint
        mf = np.zeros((T, m0.shape[0]))
        Vf = np.zeros((T, m0.shape[0], m0.shape[0]))
        S = np.zeros((T, m0.shape[0], m0.shape[0]))
        innovations = np.zeros((T, y_obs.shape[1]))  # Assuming y_obs is (T, 2)
        innovation_cov = np.zeros((T, C.shape[0], C.shape[0]))

        # Time-varying observation variance
        for k in range(ensemble_var.shape[1]):
            R[k, k] = ensemble_var[0][k]

        K_array, _ = kalman_dot(y_obs[0, :] - np.dot(C, m0), S0, C, R)
        mf[0] = m0 + K_array
        Vf[0, :] = S0 - K_array
        S[0] = S0
        innovations[0] = y_obs[0] - np.dot(C, mf[0])
        innovation_cov[0] = np.dot(C, np.dot(S0, C.T)) + R

        # Kalman filter update for subsequent time steps
        for t in range(1, T):
            # Propagate the state
            mf[t, :] = np.dot(A, mf[t - 1, :])
            S[t - 1] = np.dot(A, np.dot(Vf[t - 1, :], A.T)) + Q[k]

            if np.sum(~np.isnan(y_obs[t, :])) >= 2:  # Check if any value in y[t] is not NaN
                # Update R for time-varying observation variance
                for k in range(ensemble_var.shape[1]):
                    R[k, k] = ensemble_var[t][k]

                # Update state estimate and covariance matrix
                innovations[t] = y_obs[t, :] - np.dot(C, np.dot(A, mf[t - 1, :]))
                K_array, _ = kalman_dot(innovations[t], S[t - 1], C, R)
                mf[t, :] += K_array
                K_array, innovation_cov[t] = kalman_dot(np.dot(C, S[t - 1]), S[t - 1], C, R)
                Vf[t, :] = S[t - 1] - K_array
            else:
                Vf[t, :] = S[t - 1]

        # Store the results for the current keypoint
        mf_array[i] = mf
        Vf_array[i] = Vf
        S_array[i] = S
        innovations_array[i] = innovations
        innovation_cov_array[i] = innovation_cov
    return mf_array, Vf_array, S_array, innovations_array, innovation_cov_array


def kalman_dot(innovation, V, C, R):
    innovation_cov = R + np.dot(C, np.dot(V, C.T))
    innovation_cov_inv = np.linalg.solve(innovation_cov, innovation)
    K_array = np.dot(V, np.dot(C.T, innovation_cov_inv))
    return K_array, innovation_cov


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
    ms = np.zeros(shape=(T, mf.shape[1]))
    Vs = np.zeros(shape=(T, mf.shape[1], mf.shape[1]))
    CV = np.zeros(shape=(T - 1, mf.shape[1], mf.shape[1]))

    # Last-time smoothed posterior is equal to last-time filtered posterior
    ms[-1, :] = mf[-1, :]
    Vs[-1, :, :] = Vf[-1, :, :]
    # Smoothing steps
    for i in range(T - 2, -1, -1):
        if not np.all(np.isnan(y[i])):  # Check if all values in y[i] are not NaN
            try:
                J = np.linalg.solve(S[i], np.dot(A, Vf[i])).T
            except np.linalg.LinAlgError:
                # Skip backward pass for this timestep if matrix is singular
                continue

            Vs[i] = Vf[i] + np.dot(J, np.dot(Vs[i + 1] - S[i], J.T))
            ms[i] = mf[i] + np.dot(J, ms[i + 1] - np.dot(A, mf[i]))
            CV[i] = np.dot(Vs[i + 1], J.T)
    return ms, Vs, CV


def vectorized_backward_pass(y, mf, Vf, S, A):
    """Implements Kalman-smoothing backwards for multiple keypoints.
    Args:
        y: np.ndarray
            shape (samples, n_keypoints)
        mf: np.ndarray
            shape (samples, n_latents)
        Vf: np.ndarray
            shape (samples, n_latents, n_latents)
        S: np.ndarray
            shape (samples, n_latents, n_latents)
        A: np.ndarray
            shape (n_latents, n_latents)

    Returns:
        ms: np.ndarray
            shape (n_keypoints, samples, n_latents)
        Vs: np.ndarray
            shape (n_keypoints, samples, n_latents, n_latents)
        CV: np.ndarray
            shape (n_keypoints, samples - 1, n_latents, n_latents)
    """
    n_keypoints, T, n_coords = y.shape[0], y.shape[1], y.shape[2]
    ms = np.zeros((n_keypoints, T, mf.shape[2]))
    Vs = np.zeros((n_keypoints, T, mf.shape[2], mf.shape[2]))
    CV = np.zeros((n_keypoints, T - 1, mf.shape[2], mf.shape[2]))

    for k in range(n_keypoints):
        y_k = y[k]
        mf_k = mf[k]
        Vf_k = Vf[k]
        A_k = A[k]
        S_k = S[k]

        # Last-time smoothed posterior is equal to last-time filtered posterior
        ms[k, -1, :] = mf_k[-1, :]
        Vs[k, -1, :, :] = Vf_k[-1, :, :]
        # Smoothing steps
        for i in range(T - 2, -1, -1):
            if not np.all(np.isnan(y_k[i])):  # Check if all values in y_obs[i] are not NaN
                try:
                    J = np.linalg.solve(S_k[i], np.dot(A_k, Vf_k[i])).T
                except np.linalg.LinAlgError:
                    # Skip backward pass for this timestep if matrix is singular
                    continue

                Vs[k, i, :, :] = Vf_k[i, :, :] + np.dot(J, np.dot(Vs[k, i + 1, :, :] - S_k[i, :, :], J.T))
                ms[k, i] = mf_k[i] + np.dot(J, ms[k, i + 1] - np.dot(A_k, mf_k[i]))
                CV[k, i] = np.dot(Vs[k, i + 1], J.T)
    return ms, Vs, CV


def eks_zscore(eks_predictions, ensemble_means, ensemble_vars, min_ensemble_std=2):
    """Computes zscore between eks prediction and the ensemble for a single keypoint.
    Args:
        eks_predictions: list
            EKS prediction for each coordinate (x and y) for as single keypoint - (samples, 2)
        ensemble_means: list
            Ensemble mean for each coordinate (x and y) for as single keypoint - (samples, 2)
        ensemble_vars: string
            Ensemble var for each coordinate (x and y) for as single keypoint - (samples, 2)
        min_ensemble_std:
            Minimum std threshold to reduce the effect of low ensemble std (default 2).
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
    return z_score

def vectorized_eks_zscore(eks_predictions_array, ensemble_means, ensemble_vars, min_ensemble_std=2):
    """Computes zscore between eks prediction and the ensemble for a single keypoint.
    Args:
        eks_predictions: list
            EKS prediction for each coordinate (x and y) for as single keypoint - (samples, 2)
        ensemble_means: list
            Ensemble mean for each coordinate (x and y) for as single keypoint - (samples, 2)
        ensemble_vars: string
            Ensemble var for each coordinate (x and y) for as single keypoint - (samples, 2)
        min_ensemble_std:
            Minimum std threshold to reduce the effect of low ensemble std (default 2).
    Returns:
        z_score
            z_score for each time point - (samples, 1)
    """
    z_scores = []
    for k in range(ensemble_vars.shape[1]):  # n_keypoints
        ensemble_std = np.sqrt(
            # trace of covariance matrix - multi-d variance measure
            ensemble_vars[:, k, 0] + ensemble_vars[:, k, 1])
        num = np.sqrt(
            (eks_predictions_array[k, :, 0]
             - ensemble_means[:, k, 0]) ** 2
            + (eks_predictions_array[k, :, 1] - ensemble_means[:, k, 1]) ** 2)
        thresh_ensemble_std = ensemble_std.copy()
        thresh_ensemble_std[thresh_ensemble_std < min_ensemble_std] = min_ensemble_std
        z_scores.append(num / thresh_ensemble_std)
    print("Evaluated Z Scores.")
    return z_scores
