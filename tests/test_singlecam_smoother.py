import pytest
import numpy as np
import pandas as pd
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam


# Function to generate simulated data
def simulate_marker_data():
    np.random.seed(0)
    num_frames = 100
    num_keypoints = 5
    markers_3d_array = np.random.randn(num_frames, num_frames,
                                       num_keypoints * 3)  # Simulating 3D data for keypoints
    bodypart_list = [f'bodypart_{i}' for i in range(num_keypoints)]
    smooth_param = 0.1
    s_frames = list(range(num_frames))
    blocks = []
    ensembling_mode = 'median'
    zscore_threshold = 2
    return markers_3d_array, bodypart_list, smooth_param, s_frames, blocks, ensembling_mode, zscore_threshold


# Function to generate random likelihoods
def generate_random_likelihoods(num_frames, num_keypoints):
    return np.random.rand(num_frames, num_keypoints)


# Test function for the ensemble Kalman smoother
def test_ensemble_kalman_smoother_singlecam():
    markers_3d_array, bodypart_list, smooth_param, s_frames, blocks, ensembling_mode, zscore_threshold = simulate_marker_data()

    # Add random likelihoods to the simulated data
    likelihoods = generate_random_likelihoods(markers_3d_array.shape[0],
                                              markers_3d_array.shape[2] // 3)

    # Call the smoother function
    df_dicts, s_finals = ensemble_kalman_smoother_singlecam(
        markers_3d_array, bodypart_list, smooth_param, s_frames, blocks,
        ensembling_mode, zscore_threshold)

    # Basic checks to ensure the function runs and returns expected types
    assert isinstance(df_dicts, list), "Expected df_dicts to be a list"
    assert all(
        isinstance(d, dict) for d in df_dicts), "Expected elements of df_dicts to be dictionaries"
    assert isinstance(s_finals, (list, np.ndarray)), "Expected s_finals to be a list or an ndarray"

    # Additional checks can include verifying contents of the dataframes
    for df_dict in df_dicts:
        for key, df in df_dict.items():
            assert isinstance(df, pd.DataFrame), f"Expected {key} to be a pandas DataFrame"
            #add more detailed checks here
            assert 'likelihood' in df.columns.get_level_values(
                1), "Expected 'likelihood' in DataFrame columns"
            assert 'x_var' in df.columns.get_level_values(
                1), "Expected 'x_var' in DataFrame columns"
            assert 'y_var' in df.columns.get_level_values(
                1), "Expected 'y_var' in DataFrame columns"
            assert 'zscore' in df.columns.get_level_values(
                1), "Expected 'zscore' in DataFrame columns"
