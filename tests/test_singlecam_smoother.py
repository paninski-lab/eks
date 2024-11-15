import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import os
from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam, initialize_kalman_filter, adjust_observations
from unittest.mock import patch, MagicMock


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
            # Check for 'likelihood' in the correct level of the columns
            assert 'likelihood' in df.columns.get_level_values(
                'coords'), "Expected 'likelihood' in DataFrame columns at the 'coords' level"


def test_adjust_observations():
    # Define mock input data
    n_keypoints = 3
    keypoints_avg_dict = {
        0: np.array([1.0, 2.0, 3.0]),     # x-coordinates for keypoint 1
        1: np.array([4.0, 5.0, 6.0]),     # y-coordinates for keypoint 1
        2: np.array([0.5, 1.5, 2.5]),     # x-coordinates for keypoint 2
        3: np.array([3.5, 4.5, 5.5]),     # y-coordinates for keypoint 2
        4: np.array([2.0, 2.5, 3.0]),     # x-coordinates for keypoint 3
        5: np.array([6.0, 7.0, 8.0])      # y-coordinates for keypoint 3
    }

    # Create a mock scaled_ensemble_preds array (shape: [timepoints, n_keypoints, coordinates])
    num_samples = 3
    scaled_ensemble_preds = np.random.randn(num_samples, n_keypoints, 2)

    # Call the function
    mean_obs_dict, adjusted_obs_dict, adjusted_scaled_preds = adjust_observations(
        keypoints_avg_dict,
        n_keypoints,
        scaled_ensemble_preds
    )

    # Assertions for mean observations dictionary
    assert isinstance(mean_obs_dict, dict), "Expected mean_obs_dict to be a dictionary"
    assert len(mean_obs_dict) == 2 * n_keypoints, f"Expected {2 * n_keypoints} entries in mean_obs_dict"
    assert np.isclose(mean_obs_dict[0], np.mean(keypoints_avg_dict[0])), "Mean x-coord for keypoint 1 is incorrect"
    assert np.isclose(mean_obs_dict[1], np.mean(keypoints_avg_dict[1])), "Mean y-coord for keypoint 1 is incorrect"

    # Assertions for adjusted observations dictionary
    assert isinstance(adjusted_obs_dict, dict), "Expected adjusted_obs_dict to be a dictionary"
    assert len(adjusted_obs_dict) == 2 * n_keypoints, f"Expected {2 * n_keypoints} entries in adjusted_obs_dict"
    assert np.allclose(
        adjusted_obs_dict[0],
        keypoints_avg_dict[0] - mean_obs_dict[0]
    ), "Adjusted x-coord for keypoint 1 is incorrect"
    assert np.allclose(
        adjusted_obs_dict[1],
        keypoints_avg_dict[1] - mean_obs_dict[1]
    ), "Adjusted y-coord for keypoint 1 is incorrect"

    # Assertions for adjusted scaled ensemble predictions
    assert isinstance(adjusted_scaled_preds, jnp.ndarray), "Expected adjusted_scaled_preds to be a JAX array"
    assert adjusted_scaled_preds.shape == scaled_ensemble_preds.shape, \
        f"Expected shape {scaled_ensemble_preds.shape}, got {adjusted_scaled_preds.shape}"

    # Check that the ensemble predictions were adjusted correctly
    for i in range(n_keypoints):
        mean_x = mean_obs_dict[3 * i]
        mean_y = mean_obs_dict[3 * i + 1]
        expected_x_adjustment = scaled_ensemble_preds[:, i, 0] - mean_x
        expected_y_adjustment = scaled_ensemble_preds[:, i, 1] - mean_y
        assert np.allclose(adjusted_scaled_preds[:, i, 0], expected_x_adjustment), \
            f"Scaled ensemble preds x-coord for keypoint {i} is incorrect"
        assert np.allclose(adjusted_scaled_preds[:, i, 1], expected_y_adjustment), \
            f"Scaled ensemble preds y-coord for keypoint {i} is incorrect"

    print("Test for adjust_observations passed successfully.")


def test_initialize_kalman_filter():
    # Define test parameters
    n_samples = 10
    n_keypoints = 3

    # Generate random scaled ensemble predictions
    scaled_ensemble_preds = np.random.randn(n_samples, n_keypoints, 2)  # Shape (T, n_keypoints, 2)

    # Create a mock adjusted observations dictionary
    adjusted_obs_dict = {
        0: np.random.randn(n_samples),  # Adjusted x observations for keypoint 0
        1: np.random.randn(n_samples),  # Adjusted y observations for keypoint 0
        3: np.random.randn(n_samples),  # Adjusted x observations for keypoint 1
        4: np.random.randn(n_samples),  # Adjusted y observations for keypoint 1
        6: np.random.randn(n_samples),  # Adjusted x observations for keypoint 2
        7: np.random.randn(n_samples)   # Adjusted y observations for keypoint 2
    }

    # Run the function
    m0s, S0s, As, cov_mats, Cs, Rs, y_obs_array = initialize_kalman_filter(
        scaled_ensemble_preds, adjusted_obs_dict, n_keypoints
    )

    # Assertions to verify the function output
    assert m0s.shape == (n_keypoints, 2), f"Expected shape {(n_keypoints, 2)}, got {m0s.shape}"
    assert S0s.shape == (n_keypoints, 2, 2), f"Expected shape {(n_keypoints, 2, 2)}, got {S0s.shape}"
    assert As.shape == (n_keypoints, 2, 2), f"Expected shape {(n_keypoints, 2, 2)}, got {As.shape}"
    assert cov_mats.shape == (n_keypoints, 2, 2), f"Expected shape {(n_keypoints, 2, 2)}, got {cov_mats.shape}"
    assert Cs.shape == (n_keypoints, 2, 2), f"Expected shape {(n_keypoints, 2, 2)}, got {Cs.shape}"
    assert Rs.shape == (n_keypoints, 2, 2), f"Expected shape {(n_keypoints, 2, 2)}, got {Rs.shape}"
    assert y_obs_array.shape == (n_keypoints, n_samples, 2), f"Expected shape {(n_keypoints, n_samples, 2)}, got {y_obs_array.shape}"

    # Check that the diagonal of S0s contains non-negative values (variance cannot be negative)
    assert jnp.all(S0s[:, 0, 0] >= 0), "S0s diagonal should have non-negative variances"
    assert jnp.all(S0s[:, 1, 1] >= 0), "S0s diagonal should have non-negative variances"

    # Check that the state transition matrix is correctly initialized as identity
    expected_A = jnp.array([[1.0, 0], [0, 1.0]])
    assert jnp.allclose(As, expected_A), "State transition matrix A should be identity"

    # Check that the measurement function matrix C is correctly initialized
    expected_C = jnp.array([[1, 0], [0, 1]])
    assert jnp.allclose(Cs, expected_C), "Measurement function matrix C should be identity"

    print("Test for initialize_kalman_filter passed successfully.")


if __name__ == "__main__":
    pytest.main([__file__])
