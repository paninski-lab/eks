import jax
import numpy as np
from jax import numpy as jnp
from matplotlib import pyplot as plt
from jax import jit, vmap
import jax.scipy as jsc
from functools import partial
from jax.lax import associative_scan, psum, scan


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
    ms = np.zeros(shape=(T, mf.shape[1]))
    Vs = np.zeros(shape=(T, mf.shape[1], mf.shape[1]))
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


def eks_zscore(eks_predictions, ensemble_means, ensemble_vars, min_ensemble_std=2):
    """Computes zscore between eks prediction and the ensemble for a single keypoint.
    Args:
        eks_predictions: list
            EKS prediction for each coordinate (x and ys) for as single keypoint - (samples, 2)
        ensemble_means: list
            Ensemble mean for each coordinate (x and ys) for as single keypoint - (samples, 2)
        ensemble_vars: string
            Ensemble var for each coordinate (x and ys) for as single keypoint - (samples, 2)
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


def jax_ensemble(markers_3d_array, mode='median'):
    """
    Computes ensemble median (or mean) and variance of a 3D array of DLC marker data using JAX.

    Returns:
        ensemble_preds: np.ndarray
            shape (n_timepoints, n_keypoints, n_coordinates). ensembled predictions for each keypoint for each target
        ensemble_vars: np.ndarray
            shape (n_timepoints, n_keypoints, n_coordinates). ensembled variances for each keypoint for each target
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


def compute_covariance_matrix(ensemble_preds):
    """
    Compute the covariance matrix E for correlated noise dynamics.

    Parameters:
    ensemble_preds: A 3D array of shape (T, n_keypoints, n_coords)
                          containing the ensemble predictions.

    Returns:
    E: A 2K x 2K covariance matrix where K is the number of keypoints.
    """
    # Get the number of time steps, keypoints, and coordinates
    T, n_keypoints, n_coords = ensemble_preds.shape

    # Flatten the ensemble predictions to shape (T, 2K) where K is the number of keypoints
    flattened_preds = ensemble_preds.reshape(T, -1)

    # Compute the temporal differences
    temporal_diffs = np.diff(flattened_preds, axis=0)

    # Compute the covariance matrix of the temporal differences
    E = np.cov(temporal_diffs, rowvar=False)
    '''
    plt.imshow(E, cmap='coolwarm', interpolation='nearest', vmin=-100, vmax=100)
    plt.colorbar()  # Show a colorbar on the side
    plt.title('Heatmap of 16x16 Matrix')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    '''
    # Index covariance matrix into blocks for each keypoint
    cov_mats = []
    for i in range(n_keypoints):
        # E_block = extract_submatrix(E, i)
        E_block = [[1, 0],[0, 1]]
        cov_mats.append(E_block)
    cov_mats = jnp.array(cov_mats)
    return cov_mats


def extract_submatrix(Qs, i, submatrix_size=2):
    # Compute the start indices for the submatrix
    i_q = 2 * i
    start_indices = (i_q, i_q)

    # Use jax.lax.dynamic_slice to extract the submatrix
    submatrix = jax.lax.dynamic_slice(Qs, start_indices, (submatrix_size, submatrix_size))

    return submatrix


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


# --------------------------------------------------------------------
# JAX functions (will rename once other scripts all use these funcs)
# --------------------------------------------------------------------


def first_filtering_element(C, A, Q, R, m0, P0, y):
    # model.F = A, model.H = C,
    S = C @ Q @ C.T + R
    CF, low = jsc.linalg.cho_factor(S)  # note the jsc

    m1 = A @ m0
    P1 = A @ P0 @ A.T + Q
    S1 = C @ P1 @ C.T + R
    K1 = jsc.linalg.solve(S1, C @ P1, assume_a='pos').T  # note the jsc

    A_updated = jnp.zeros_like(A)
    b = m1 + K1 @ (y - C @ m1)
    C_updated = P1 - K1 @ S1 @ K1.T

    # note the jsc
    eta = A.T @ C.T @ jsc.linalg.cho_solve((CF, low), y)
    J = A.T @ C.T @ jsc.linalg.cho_solve((CF, low), C @ A)
    return A_updated, b, C_updated, J, eta


def generic_filtering_element(C, A, Q, R, y):
    S = C @ Q @ C.T + R
    CF, low = jsc.linalg.cho_factor(S)  # note the jsc
    K = jsc.linalg.cho_solve((CF, low), C @ Q).T  # note the jsc
    A_updated = A - K @ C @ A 
    b = K @ y
    C_updated = Q - K @ C @ Q

    # note the jsc
    eta = A.T @ C.T @ jsc.linalg.cho_solve((CF, low), y)
    J = A.T @ C.T @ jsc.linalg.cho_solve((CF, low), C @ A)
    return A_updated, b, C_updated, J, eta
    
def make_associative_filtering_elements(C, A, Q, R, m0, P0, observations):
    first_elems = first_filtering_element(C, A, Q, R, m0, P0, observations[0])
    generic_elems = vmap(lambda o: generic_filtering_element(C, A, Q, R, o))(observations[1:])
    return tuple(jnp.concatenate([jnp.expand_dims(first_e, 0), gen_es]) 
                 for first_e, gen_es in zip(first_elems, generic_elems))

@partial(vmap)
def filtering_operator(elem1, elem2):
    # # note the jsc everywhere
    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2
    dim = A1.shape[0]
    I = jnp.eye(dim)  # note the jnp

    I_C1J2 = I + C1 @ J2
    temp = jsc.linalg.solve(I_C1J2.T, A2.T).T
    A = temp @ A1
    b = temp @ (b1 + C1 @ eta2) + b2
    C = temp @ C1 @ A2.T + C2

    I_J2C1 = I + J2 @ C1
    temp = jsc.linalg.solve(I_J2C1.T, A1).T

    eta = temp @ (eta2 - J2 @ b1) + eta1
    J = temp @ J2 @ A1 + J1

    return A, b, C, J, eta


def pkf(y, m0, cov0, A, Q, C, R):
    initial_elements = make_associative_filtering_elements(C, A, Q, R, m0, cov0, y)
    final_elements = associative_scan(filtering_operator, initial_elements)
    return final_elements
    
pkf_func = jit(pkf)


def get_kalman_means(A_scan, b_scan, m0):
    """
    Computes the Kalman mean at a single timepoint, the result is: 
    A_scan @ m0 + b_scan

    Returned shape: (state_dimension, 1)
    """
    return A_scan @ jnp.expand_dims(m0, axis = 1) + jnp.expand_dims(b_scan, axis = 1)

def get_kalman_variances(C):
    return C

def get_next_cov(A, C, Q, R, filter_cov, filter_mean):
    """
    Given the moments of p(x_t | y_1, ..., y_t) (normal filter distribution), compute the moments of the distribution for: 
    p(y_{t+1} | y_1, ..., y_t)

    Params: 
        A (np.ndarray): Shape (state_dimension, state_dimension) Process coeff matrix
        C (np.ndarray): Shape (obs_dimension, state_dimension) Observation coeff matrix
        Q (np.ndarray): Shape (state_dimension, state_dimension). Process noise covariance matrix. 
        R (np.ndarray): Shape (obs_dimension, obs_dimension). Observation noise covariance matrix. 
        filter_cov (np.ndarray). Shape (state_dimension, state_dimension). Filtered covariance
        filter_mean (np.ndarray). Shape (state_dimension, 1). Filter mean 

    Returns: 
        mean (np.ndarray). Shape (obs_dimension, 1)
        cov (np.ndarray). Shape (obs_dimension, obs_dimension). 
    """
    mean = C @ A @ filter_mean
    cov = C @ (A @ filter_cov @ A.T + Q) @ C.T + R
    return mean, cov

def compute_marginal_nll(value, mean, covariance):
    return -1*jax.scipy.stats.multivariate_normal.logpdf(value, mean, covariance)

def parallel_loss_single(A_scan, b_scan, C_scan, A, C, Q, R, next_observation, m0):
    curr_mean = get_kalman_means(A_scan, b_scan, m0)
    curr_cov = get_kalman_variances(C_scan) #Placeholder; just returns identity

    next_mean, next_cov = get_next_cov(A, C, Q, R, curr_cov, curr_mean)
    return jnp.squeeze(curr_mean), curr_cov, compute_marginal_nll(jnp.squeeze(next_observation), jnp.squeeze(next_mean), next_cov)

parallel_loss_func_vmap = jit(vmap(parallel_loss_single, in_axes = (0, 0, 0, None, None, None, None, 0, None), out_axes = (0, 0, 0)))

@partial(jit)
def y1_given_x0_nll(C, A, Q, R, m0, cov0, obs):
    y1_predictive_mean = C @ A @ jnp.expand_dims(m0, axis = 1)
    y1_predictive_cov = C @ (A @ cov0 @ A.T + Q) @ C.T + R
    addend = -1*jax.scipy.stats.multivariate_normal.logpdf(obs, jnp.squeeze(y1_predictive_mean), y1_predictive_cov)
    return addend
    

def pkf_and_loss(y, m0, cov0, A, Q, C, R):
    A_scan, b_scan, C_scan, _, _ = pkf_func(y, m0, cov0, A, Q, C, R)

    #Gives us the NLL for p(y_i | y_1, ..., y_{i-1}) for i > 1. Need to use the parallel scan outputs for this. i = 1 handled below
    filtered_states, filtered_covariances, losses = parallel_loss_func_vmap(A_scan[:-1], b_scan[:-1], C_scan[:-1], A, C, Q, R, y[1:], m0)
    

    #Gives us the NLL for p_y(y_1 | x_0)
    addend = y1_given_x0_nll(C, A, Q, R, m0, cov0, y[0])
    
    final_mean = get_kalman_means(A_scan[-1], b_scan[-1], m0).T
    final_covariance = jnp.expand_dims(get_kalman_variances(C_scan[-1]), axis = 0)
    filtered_states = jnp.concatenate([filtered_states, final_mean], axis = 0)
    filtered_variances = jnp.concatenate([filtered_covariances, final_covariance], axis = 0)
    return filtered_states, filtered_variances, jnp.sum(losses) + addend



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
    V_t = V_pred - jnp.dot(K, jnp.dot(C, V_pred))

    nll_current = single_timestep_nll(innovation, innovation_cov)
    nll_net = nll_net + nll_current

    return (m_t, V_t, A, Q, C, R, nll_net), (m_t, V_t, nll_current)


    

#Always run the sequential filter on CPU because the GPU will deploy individual kernels for each scan iteration, very slow. 
@partial(jit, backend='cpu')
def jax_forward_pass(y, m0, cov0, A, Q, C, R):
    """
    Kalman Filter for a single keypoint (can be vectorized using vmap for handling multiple keypoints in parallel)
    Parameters:
        y (np.ndarray): Shape (num_timepoints, observation_dimension). 
        m0 (np.ndarray): Shape (state_dimension,). Initial state of system. 
        cov0 (np.ndarray): Shape (state_dimension, state_dimension). Initial covariance of state variable. 
        A (np.ndarray): Shape (state_dimension, state_dimension). Process transition matrix.
        Q (np.ndarray): Shape (state_dimension, state_dimension). Process noise covariance matrix. 
        C (np.ndarray): Shape (observation_dimension, state_dimension). Observation coefficient matrix. 
        R (np.ndarray): Shape (observation_dimension, observation_dimension). Observation noise covariance matrix. 

    Returns:
        mfs (np.ndarray): Shape (num_timepoints, state_dimension). Mean filter state at each timepoint.
        Vfs (np.ndarray): Shape (num_timepoints, state_dimension, state_dimension). Covariance for each filtered state estimate.
        nll_net (np.ndarray): Shape (1,). Negative log likelihood observations -log (p(y_1, ..., y_T)) where T is total number of timepoints. 
    """
    # Initialize carry
    carry = (m0, cov0, A, Q, C, R, 0)
    carry, outputs = jax.lax.scan(kalman_filter_step, carry, y)
    mfs, Vfs, _ = outputs
    nll_net = carry[-1]
    return mfs, Vfs, nll_net


def kalman_smoother_step(carry, X):
    m_ahead_smooth, v_ahead_smooth, A, Q = carry
    m_curr_filter, v_curr_filter = X[0], X[1]

    # Compute the smoother gain
    ahead_predict = jnp.dot(A, m_curr_filter)
    ahead_cov = jnp.dot(A, jnp.dot(v_curr_filter, A.T)) + Q

    smoothing_gain = jsc.linalg.solve(ahead_cov, jnp.dot(A, v_curr_filter.T)).T
    smoothed_state = m_curr_filter + jnp.dot(smoothing_gain, m_ahead_smooth - m_curr_filter)
    smoothed_cov = v_curr_filter + jnp.dot(jnp.dot(smoothing_gain, m_ahead_smooth - ahead_cov), smoothing_gain.T)

    return (smoothed_state, smoothed_cov, A, Q), (smoothed_state, smoothed_cov)

@partial(jit, backend='cpu')
def jax_backward_pass(mfs, Vfs, A, Q):
    """
    Runs the kalman smoother given the filtered values
    Parameters: 
        mfs (np.ndarray). Shape (num_timepoints, state_dimension). The kalman-filtered means of the data. 
        Vfs (np.ndarray). Shape (num_timepoints, state_dimension, state_dimension). The kalman-filtered covariance matrix of the state vector at each time point.
        A (np.ndarray). Shape (state_dimension, state_dimension). The process transition matrix
        Q (np.ndarray). Shape (state_dimension, state_dimension). The covariance of the process noise. 
    Returns:
        smoothed_states (np.ndarray): Shape (num_timepoints, state_dimension). The smoothed estimates for the state vector starting at the first timepoint where observations are possible. 
        smoothed_state_covariances (np.ndarray): Shape (num_timepoints, state_dimension, state_dimension). 
    """
    carry = (mfs[-1], Vfs[-1], A, Q)

    # Reverse scan over the time steps
    _, outputs = jax.lax.scan(
        kalman_smoother_step,
        carry,
        [mfs[:-1], Vfs[:-1]],
        reverse=True
    )
    
    smoothed_states, smoothed_state_covariances = outputs
    smoothed_states = jnp.append(smoothed_states, jnp.expand_dims(mfs[-1], 0), 0)
    smoothed_state_covariances = jnp.append(smoothed_state_covariances, jnp.expand_dims(Vfs[-1], 0), 0)
    return smoothed_states, smoothed_state_covariances


def single_timestep_nll(innovation, innovation_cov):
    epsilon=1e-6
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

single_timestep_nll_vmap = vmap(single_timestep_nll, in_axes=(0, 0))

@partial(jit)
def jax_compute_nll(innovations, innovation_covs):
    """
    Computes the negative log likelihood (NLL) for the EKS prediction in a parallelized manner using JAX.
    Args:
        
    """
    # Apply the single_timestep_nll function across all time steps using vmap
    nll_values = single_timestep_nll_vmap(innovations, innovation_covs)

    # Sum all the NLL increments to get the total NLL
    nll = jnp.sum(nll_values)
    return nll, nll_values
