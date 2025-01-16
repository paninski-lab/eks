import numpy as np
import pandas as pd

from eks.multicam_smoother import ensemble_kalman_smoother_multicam


def test_ensemble_kalman_smoother_multicam():
    """Test the basic functionality of ensemble_kalman_smoother_multicam."""
    # Mock inputs
    keypoint_names = ['kp1', 'kp2']
    columns = [f'{kp}_{coord}' for kp in keypoint_names for coord in ['x', 'y', 'likelihood']]
    markers_list_cameras = [
        [
            [
                pd.DataFrame(np.random.randn(100, len(columns)), columns=columns),
                pd.DataFrame(np.random.randn(100, len(columns)), columns=columns),
            ] for _ in range(2)
        ] for _ in range(2)]
    camera_names = ['cam1', 'cam2']

    smooth_param = 0.1
    quantile_keep_pca = 95
    s_frames = None
    zscore_threshold = 2

    # Run the smoother
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        markers_list=markers_list_cameras,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='median',
    )

    # Assertions
    assert isinstance(camera_dfs, list), "Expected output to be a dictionary"
    assert len(camera_dfs) == len(camera_names), \
        f"Expected {len(camera_names)} entries in camera_dfs, got {len(camera_dfs)}"
    assert isinstance(smooth_params_final, float), \
        f"Expected smooth_param_final to be a float, got {type(smooth_params_final)}"
    assert smooth_params_final == smooth_param, \
        f"Expected smooth_param_final to match input smooth_param ({smooth_param}), " \
        f"got {smooth_params_final}"


def test_ensemble_kalman_smoother_multicam_no_smooth_param():
    """Test ensemble_kalman_smoother_multicam with no smooth_param provided."""
    # Mock inputs
    keypoint_names = ['kp1', 'kp2']
    columns = [f'{kp}_{coord}' for kp in keypoint_names for coord in ['x', 'y', 'likelihood']]
    markers_list_cameras = [
        [
            [
                pd.DataFrame(np.random.randn(100, len(columns)), columns=columns),
                pd.DataFrame(np.random.randn(100, len(columns)), columns=columns),
            ] for _ in range(2)
        ] for _ in range(2)]
    camera_names = ['cam1', 'cam2']

    quantile_keep_pca = 90
    s_frames = [(0, 10)]

    # Run the smoother without providing smooth_param
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        markers_list=markers_list_cameras,
        keypoint_names=keypoint_names,
        smooth_param=None,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='median',
    )

    # Assertions
    assert smooth_params_final is not None, "Expected smooth_param_final to be not None"
    assert isinstance(smooth_params_final, float), \
        f"Expected smooth_param_final to be a float, got {type(smooth_params_final)}"
