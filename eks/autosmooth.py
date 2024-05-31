import numpy as np
from scipy.optimize import minimize

from eks.core import forward_pass, backward_pass, vectorized_forward_pass, vectorized_backward_pass

''' Contains all auto smoothing parameter selection functions and variants '''


def subset_by_frames(y, s_frames):
    # Create an empty list to store arrays
    result = []

    for frame in s_frames:
        # Unpack the frame, setting defaults for empty start or end
        start, end = frame
        # Default start to 0 if not specified (and adjust for zero indexing)
        start = start - 1 if start is not None else 0
        # Default end to the length of ys if not specified
        end = end if end is not None else len(y)

        # Validate the keys
        if start < 0 or end > len(y) or start >= end:
            raise ValueError(f"Index range ({start + 1}, {end}) "
                             f"is out of bounds for the list of length {len(y)}.")

        # Use numpy slicing to preserve the data structure
        result.append(y[start:end])

    # Concatenate all slices into a single numpy array
    return np.concatenate(result)


def compute_initial_guesses(ensemble_vars):

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
    print(f'Initial guess: {std_dev_guess}')
    return std_dev_guess


def compute_nll(innovations, innovation_covs, epsilon=1e-6):
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


def vectorized_compute_nll(innovations, innovation_covs, epsilon=1e-6):
    n_keypoints, T, n_coords = innovations.shape[0], innovations.shape[1], innovations.shape[2]
    nll = np.zeros(n_keypoints)
    nll_values_array = [np.array([]) for _ in range(n_keypoints)]
    c = np.log(2 * np.pi) * n_coords  # The Gaussian normalization constant part
    for k in range(n_keypoints):
        nll_values = []
        for t in range(T):
            # Check if any value in innovations[k, t] is not NaN
            if not np.any(np.isnan(innovations[k, t])):
                # Regularize the innovation covariance matrix by adding epsilon to the diagonal
                reg_innovation_cov = innovation_covs[k, t] + epsilon * np.eye(n_coords)

                # Compute the log determinant of the regularized covariance matrix
                log_det_S = np.log(np.abs(np.linalg.det(reg_innovation_cov)) + epsilon)
                solved_term = np.linalg.solve(reg_innovation_cov, innovations[k, t])
                quadratic_term = np.dot(innovations[k, t], solved_term)

                # Compute the NLL increment for time step t
                nll_increment = 0.5 * np.abs((log_det_S + quadratic_term + c))
                nll_values.append(nll_increment)
                nll[k] += nll_increment

        nll_values_array[k] = np.array(nll_values)

    return nll, nll_values_array


def singlecam_multicam_optimize_and_smooth(
        cov_matrix, y, m0, s0, C, A, R, ensemble_vars,
        s_frames=[(1, 2000)],
        smooth_param=None):
    # Optimize smooth_param
    if smooth_param is None:
        guess = compute_initial_guesses(ensemble_vars)

        # Update xatol during optimization
        def callback(xk):
            # Update xatol based on the current solution xk
            xatol = np.log(np.abs(xk)) * 0.01

            # Update the options dictionary with the new xatol value
            options['xatol'] = xatol

        # Initialize options with initial xatol
        options = {'xatol': np.log(guess)}

        # Unpack s_frames
        y_shortened = subset_by_frames(y, s_frames)

        # Minimize negative log likelihood
        smooth_param = minimize(
            singlecam_multicam_smooth_min,
            x0=guess,  # initial smooth param guess
            args=(cov_matrix, y_shortened, m0, s0, C, A, R, ensemble_vars),
            method='Nelder-Mead',
            options=options,
            callback=callback,  # Pass the callback function
            bounds=[(0, None)]
        )
        smooth_param = round(smooth_param.x[0], 5)
        print(f'Optimal at s={smooth_param}')

    # Final smooth with optimized s
    ms, Vs, nll, nll_values = singlecam_multicam_smooth_final(
        cov_matrix, smooth_param, y, m0, s0, C, A, R, ensemble_vars)

    return smooth_param, ms, Vs, nll, nll_values


def vectorized_singlecam_multicam_optimize_and_smooth(
        cov_mats, ys, m0s, s0s, Cs, As, Rs, ensemble_vars,
        s_frames=[(1, 2000)],
        smooth_param=None):
    n_keypoints = ys.shape[0]
    s_finals = []
    # Optimize smooth_param
    if smooth_param is None:
        guesses = []
        y_array_shortened = np.zeros(ys.shape)
        for k in range(n_keypoints):
            guesses.append(compute_initial_guesses(ensemble_vars[:, k, :]))
            # Unpack s_frames
            y_array_shortened[k] = subset_by_frames(ys[k], s_frames)

        # Update xatol during optimization
        def callback(xk):
            # Update xatol based on the current solution xk
            xatol = np.log(np.abs(xk)) * 0.01

            # Update the options dictionary with the new xatol value
            options['xatol'] = xatol

        # Initialize options with initial xatol
        options = {'xatol': np.log(guesses[k])}

        # Minimize negative log likelihood
        s_finals = minimize(
            vectorized_singlecam_multicam_smooth_min,
            x0=guesses,  # initial smooth param guess
            args=(cov_mats, y_array_shortened, m0s,
                  s0s, Cs, As, Rs, ensemble_vars),
            method='Nelder-Mead',
            options=options,
            callback=callback,  # Pass the callback function
            bounds=[(0, None)]
        )
        s_finals = s_finals.x
        print(f'Optimal at s={s_finals}')
    else:
        s_finals = [smooth_param]

    # Final smooth with optimized s
    ms, Vs, nll, nll_values = vectorized_singlecam_multicam_smooth_final(
        cov_mats, s_finals,
        ys, m0s, s0s, Cs, As, Rs, ensemble_vars)

    return s_finals, ms, Vs, nll, nll_values


def pupil_optimize_and_smooth(
        y, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var,
        s_frames=[(1, 2000)],
        smooth_params=[None, None]):
    # Optimize smooth_param
    if smooth_params[0] is None or smooth_params[1] is None:

        # Unpack s_frames
        y_shortened = subset_by_frames(y, s_frames)

        # Minimize negative log likelihood
        smooth_params = minimize(
            pupil_smooth_min,  # function to minimize
            x0=[1, 1],
            args=(y_shortened, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var),
            method='Nelder-Mead',
            tol=0.002,
            bounds=[(0, 1), (0, 1)]  # bounds for each parameter in smooth_params
        )
        smooth_params = [round(smooth_params.x[0], 5), round(smooth_params.x[1], 5)]
        print(f'Optimal at diameter_s={smooth_params[0]}, com_s={smooth_params[1]}')

    # Final smooth with optimized s
    ms, Vs, nll, nll_values = pupil_smooth_final(
        y, smooth_params, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var)

    return smooth_params, ms, Vs, nll, nll_values


def singlecam_multicam_smooth_final(cov_matrix, smooth_param, y, m0, S0, C, A, R, ensemble_vars):
    # Adjust Q based on smooth_param and cov_matrix
    Q = smooth_param * cov_matrix
    # Run filtering and smoothing with the current smooth_param
    mf, Vf, S, innovs, innov_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    ms, Vs, CV = backward_pass(y, mf, Vf, S, A)
    # Compute the negative log-likelihood based on innovations and their covariance
    nll, nll_values = compute_nll(innovs, innov_cov)
    return ms, Vs, nll, nll_values


def vectorized_singlecam_multicam_smooth_final(
        cov_mats, s_finals, y, m0, S0, C, A, R, ensemble_vars):
    Q = []
    for s in s_finals:
        Q.append(s * cov_mats)
    mf, Vf, S, innovs, innov_cov = vectorized_forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    ms, Vs, CV = vectorized_backward_pass(y, mf, Vf, S, A)
    nll, nll_values = vectorized_compute_nll(innovs, innov_cov)
    return ms, Vs, nll, nll_values


def singlecam_multicam_smooth_min(cov_matrix, smooth_param, y, m0, S0, C, A, R, ensemble_vars):
    # Adjust Q based on smooth_param and cov_matrix
    Q = smooth_param * cov_matrix
    # Run filtering with the current smooth_param
    mf, Vf, S, innovs, innov_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    # Compute the negative log-likelihood based on innovations and their covariance
    nll, nll_values = compute_nll(innovs, innov_cov)
    return nll


def vectorized_singlecam_multicam_smooth_min(
        s_finals, cov_mats, y, m0, S0, C, A, R, ensemble_vars):
    Q = []
    for s in s_finals:
        Q.append(s * cov_mats)
    mf, Vf, S, innovs, innov_cov = vectorized_forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    nll, _ = vectorized_compute_nll(innovs, innov_cov)
    return sum(nll)


def pupil_smooth_final(y, smooth_params, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var):
    # Construct state transition matrix
    diameter_s = smooth_params[0]
    com_s = smooth_params[1]
    A = np.asarray([
        [diameter_s, 0, 0],
        [0, com_s, 0],
        [0, 0, com_s]
    ])
    # cov_matrix
    Q = np.asarray([
        [diameters_var * (1 - (A[0, 0] ** 2)), 0, 0],
        [0, x_var * (1 - A[1, 1] ** 2), 0],
        [0, 0, y_var * (1 - (A[2, 2] ** 2))]
    ])
    # Run filtering and smoothing with the current smooth_param
    mf, Vf, S, innovs, innov_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)
    ms, Vs, CV = backward_pass(y, mf, Vf, S, A)
    # Compute the negative log-likelihood based on innovations and their covariance
    nll, nll_values = compute_nll(innovs, innov_cov)
    return ms, Vs, nll, nll_values


def pupil_smooth_min(smooth_params, y, m0, S0, C, R, ensemble_vars, diameters_var, x_var, y_var):
    # Construct As
    diameter_s, com_s = smooth_params[0], smooth_params[1]
    A = np.array([
        [diameter_s, 0, 0],
        [0, com_s, 0],
        [0, 0, com_s]
    ])

    # Construct cov_matrix Q
    Q = np.array([
        [diameters_var * (1 - (A[0, 0] ** 2)), 0, 0],
        [0, x_var * (1 - A[1, 1] ** 2), 0],
        [0, 0, y_var * (1 - (A[2, 2] ** 2))]
    ])

    # Run filtering with the current smooth_param
    mf, Vf, S, innovs, innov_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)

    # Compute the negative log-likelihood
    nll, nll_values = compute_nll(innovs, innov_cov)

    return nll
