import numpy as np
import pandas as pd
from eks.utils import make_dlc_pandas_index
from eks.core import ensemble, eks_zscore, optimize_smoothing_params, \
    filter_smooth_nll


# -----------------------
# funcs for single-view
# -----------------------
def ensemble_kalman_smoother_single_view(
        markers_list, keypoint_ensemble, smooth_param, ensembling_mode='median',
        zscore_threshold=2, verbose=False):
    """ Use an identity observation matrix and smoothes by adjusting the smoothing parameter in the
    state-covariance matrix.

    Parameters
    ----------
    markers_list : list of list of pd.DataFrames
        each list element is a list of dataframe predictions from one ensemble member.
    keypoint_ensemble : str
        the name of the keypoint to be ensembled and smoothed
    smooth_param : float
        ranges from .01-20 (smaller values = more smoothing)
    ensembling_mode:
        the function used for ensembling ('mean', 'median', or 'confidence_weighted_mean')
    zscore_threshold:
        Minimum std threshold to reduce the effect of low ensemble std on a zscore metric
        (default 2).
    verbose: bool
        If True, progress will be printed for the user.
    Returns
    -------

    Returns
    -------
    dict
        keypoint_df: dataframe containing smoothed markers for one keypoint; same format as input
        dataframes
    """

    # --------------------------------------------------------------
    # interpolate right cam markers to left cam timestamps
    # --------------------------------------------------------------
    keys = [keypoint_ensemble + '_x', keypoint_ensemble + '_y']
    x_key = keys[0]
    y_key = keys[1]

    # compute ensemble median
    ensemble_preds, ensemble_vars, ensemble_stacks, keypoints_mean_dict, keypoints_var_dict, \
        keypoints_stack_dict = ensemble(markers_list, keys, mode=ensembling_mode)
    mean_x_obs = np.nanmean(keypoints_mean_dict[x_key])
    mean_y_obs = np.nanmean(keypoints_mean_dict[y_key])
    x_t_obs, y_t_obs = \
        keypoints_mean_dict[x_key] - mean_x_obs, keypoints_mean_dict[y_key] - mean_y_obs
    # z_t_obs = np.vstack((x_t_obs, y_t_obs))  # latent variables - true x and y

    # ------ Set values for kalman filter ------
    m0 = np.asarray([0.0, 0.0])  # initial state: mean
    S0 = np.asarray([[np.nanvar(x_t_obs), 0.0], [0.0 , np.nanvar(y_t_obs)]])  # diagonal: var

    A = np.asarray([[1.0, 0], [0, 1.0]])  # state-transition matrix,
    cov_matrix = np.asarray([[1, 0], [0, 1]])  # state covariance matrix; smaller = more smoothing
    C = np.asarray([[1, 0], [0, 1]])  # Measurement function
    R = np.eye(2)  # placeholder diagonal matrix for ensemble variance

    scaled_ensemble_preds = ensemble_preds.copy()
    scaled_ensemble_preds[:, 0] -= mean_x_obs
    scaled_ensemble_preds[:, 1] -= mean_y_obs

    y_obs = scaled_ensemble_preds

    '''
    if verbose:
        print(f"filtering {keypoint_ensemble}...")
    mf, Vf, S = filtering_pass(y_obs, m0, S0, C, R, A, Q, ensemble_vars)
    if verbose:
        print("done filtering")
    y_m_filt = np.dot(C, mf.T).T
    y_v_filt = np.swapaxes(np.dot(C, np.dot(Vf, C.T)), 0, 1)

    # Do the smoothing step
    if verbose:
        print(f"smoothing {keypoint_ensemble}...")
    ms, Vs, _ = smooth_backward(y_obs, mf, Vf, S, A, Q, C)
    if verbose:
        print("done smoothing")
    # compute NLL
    nll = compute_nll_2(y_obs, mf, S, C)
    nll_values = compute_nll_2_steps(y_obs, mf, S, C)
    '''


    # Call functions from ensemble_kalman to optimize smooth_param before filtering and smoothing
    if smooth_param is None:
        smooth_param_final = \
            optimize_smoothing_params(cov_matrix, y_obs, m0, S0, C, A, R, ensemble_vars)
    else:
        smooth_param_final = smooth_param
    ms, Vs, nll, nll_values = \
        filter_smooth_nll(cov_matrix, smooth_param_final, y_obs, m0, S0, C, A, R, ensemble_vars)
    print(f"NLL is {nll} for {keypoint_ensemble}, smooth_param={smooth_param_final}")

    # Smoothed posterior over y
    y_m_smooth = np.dot(C, ms.T).T
    y_v_smooth = np.swapaxes(np.dot(C, np.dot(Vs, C.T)), 0, 1)

    # compute zscore for EKS to see how it deviates from the ensemble
    eks_predictions = y_m_smooth.copy()
    eks_predictions = \
        np.asarray([eks_predictions.T[0] + mean_x_obs, eks_predictions.T[1] + mean_y_obs]).T
    zscore = \
        eks_zscore(eks_predictions, ensemble_preds, ensemble_vars,
                   min_ensemble_std=zscore_threshold)

    # --------------------------------------
    # final cleanup
    # --------------------------------------
    pdindex = make_dlc_pandas_index([keypoint_ensemble],
                                    labels=["x", "y", "likelihood", "x_var", "y_var", "zscore"])
    var = np.empty(y_m_smooth.T[0].shape)
    var[:] = np.nan
    pred_arr = np.vstack([
        y_m_smooth.T[0] + mean_x_obs,
        y_m_smooth.T[1] + mean_y_obs,
        var,
        y_v_smooth[:, 0, 0],
        y_v_smooth[:, 1, 1],
        zscore,
    ]).T
    df = pd.DataFrame(pred_arr, columns=pdindex)
    return {keypoint_ensemble + '_df': df}, smooth_param_final, nll_values


'''
Plotting NLL traces (paste in before final cleanup)
    # Plot nll values against time
    plt.plot(range(len(nll_values)), nll_values)
    plt.xlabel('Time Step')
    plt.ylabel('Negative Log Likelihood (nll)')
    plt.title(f'Negative Log Likelihood vs Time for IBL Pupil s={smooth_param}')
    plt.grid(True)

    # Save the plot as a PDF file
    plt.savefig('nll_plot.pdf')

    plt.show()
'''
