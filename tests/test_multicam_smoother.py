import numpy as np
from sklearn.decomposition import PCA

from eks.marker_array import MarkerArray
from eks.multicam_smoother import ensemble_kalman_smoother_multicam, inflate_variance
from eks.utils import center_predictions


def test_ensemble_kalman_smoother_multicam():
    """Test the basic functionality of ensemble_kalman_smoother_multicam."""

    # Mock inputs
    keypoint_names = ['kp1', 'kp2']
    data_fields = ['x', 'y', 'likelihood']
    n_models = 5
    n_cameras = 2
    n_frames = 100
    num_fields = len(data_fields)

    # Create mock MarkerArray with explicit data_fields
    dump_array = np.random.randn(n_models, n_cameras, n_frames, len(keypoint_names), num_fields)
    markers_array = np.random.randn(n_models, n_cameras, n_frames, len(keypoint_names), num_fields)
    marker_array = MarkerArray(markers_array, data_fields=data_fields)

    camera_names = ['cam1', 'cam2']
    smooth_param = 0.1
    quantile_keep_pca = 95
    s_frames = None

    # ---------------------------------------------------
    # Run the smoother
    # ---------------------------------------------------
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='median',
        inflate_vars=False,
    )
    assert isinstance(camera_dfs, list), "Expected output to be a list"
    assert len(camera_dfs) == len(camera_names), \
        f"Expected {len(camera_names)} entries in camera_dfs, got {len(camera_dfs)}"
    assert isinstance(smooth_params_final, np.ndarray), \
        f"Expected smooth_param_final to be an array, got {type(smooth_params_final)}"
    assert smooth_params_final == smooth_param, \
        f"Expected smooth_param_final to match input smooth_param ({smooth_param}), " \
        f"got {smooth_params_final}"

    # ---------------------------------------------------
    # Run with variance inflation
    # ---------------------------------------------------
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='median',
        inflate_vars=True,
        inflate_vars_kwargs={'likelihood_threshold': None}
    )
    # Assertions
    assert isinstance(camera_dfs, list), "Expected output to be a list"
    assert len(camera_dfs) == len(camera_names), \
        f"Expected {len(camera_names)} entries in camera_dfs, got {len(camera_dfs)}"
    assert isinstance(smooth_params_final, np.ndarray), \
        f"Expected smooth_param_final to be an array, got {type(smooth_params_final)}"
    assert smooth_params_final == smooth_param, \
        f"Expected smooth_param_final to match input smooth_param ({smooth_param}), " \
        f"got {smooth_params_final}"

    # ---------------------------------------------------
    # Run with variance inflation + more maha kwargs
    # ---------------------------------------------------
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='mean',
        inflate_vars=True,
        inflate_vars_kwargs={
            'loading_matrix': np.random.randn(2 * len(camera_names), 3),
            'mean': np.random.randn(2 * len(camera_names))
        }
    )
    assert isinstance(camera_dfs, list), "Expected output to be a list"
    assert len(camera_dfs) == len(camera_names), \
        f"Expected {len(camera_names)} entries in camera_dfs, got {len(camera_dfs)}"
    assert isinstance(smooth_params_final, np.ndarray), \
        f"Expected smooth_param_final to be an array, got {type(smooth_params_final)}"
    assert smooth_params_final == smooth_param, \
        f"Expected smooth_param_final to match input smooth_param ({smooth_param}), " \
        f"got {smooth_params_final}"

    # ---------------------------------------------------
    # Run with variance inflation + more maha kwargs
    # ---------------------------------------------------
    # create true 3d data (non-centered)
    np.random.seed(0)
    data = 5.0 + np.random.randn(n_frames * len(keypoint_names), 2 * n_cameras)
    pca = PCA(n_components=3).fit(data)
    data_3d = pca.transform(data)
    data_reproj = pca.inverse_transform(data_3d)
    markers_array = np.random.randn(5, n_cameras, n_frames, len(keypoint_names), num_fields)
    for k in range(len(keypoint_names)):
        for c in range(n_cameras):
            xs = data_reproj[n_frames * k: n_frames * (k + 1), 2 * c]
            ys = data_reproj[n_frames * k: n_frames * (k + 1), 2 * c + 1]
            for m in range(n_models):
                markers_array[m, c, :, k, 0] = 0.001 * np.random.randn(*xs.shape) + xs
                markers_array[m, c, :, k, 1] = 0.001 * np.random.randn(*ys.shape) + ys
    markers_array[..., 2] = 1.0
    marker_array = MarkerArray(markers_array, data_fields=data_fields)
    # run with variance inflation
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=marker_array,
        keypoint_names=keypoint_names,
        smooth_param=smooth_param,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        inflate_vars=True,
        inflate_vars_kwargs={
            'loading_matrix': pca.components_.T,
            'mean': pca.mean_,
        }
    )
    # ensemble variance should be very small since the data is a slightly noisy copy of a template
    for df in camera_dfs:
        mask = df.columns.get_level_values("coords").isin(["x_ens_var", "y_ens_var"])
        assert np.allclose(df.iloc[:, mask], 0, atol=1e-4), "should have zero ensemble variance"


def test_ensemble_kalman_smoother_multicam_no_smooth_param():
    """Test ensemble_kalman_smoother_multicam with no smooth_param provided."""
    # Mock inputs
    keypoint_names = ['kp1', 'kp2']
    data_fields = ['x', 'y', 'likelihood']
    n_cameras = 2
    n_frames = 100
    num_fields = len(data_fields)

    # Create mock MarkerArray with explicit data_fields
    markers_array = np.random.randn(3, n_cameras, n_frames, len(keypoint_names), num_fields)
    markerArray = MarkerArray(markers_array, data_fields=data_fields)  # Ensure data_fields is set

    camera_names = ['cam1', 'cam2']
    quantile_keep_pca = 90
    s_frames = [(0, 100)]

    # Run the smoother without providing smooth_param
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        marker_array=markerArray,
        keypoint_names=keypoint_names,
        smooth_param=None,
        quantile_keep_pca=quantile_keep_pca,
        camera_names=camera_names,
        s_frames=s_frames,
        avg_mode='median',
        inflate_vars=False,
    )

    # Assertions
    assert smooth_params_final is not None, "Expected smooth_param_final to be not None"
    assert isinstance(smooth_params_final, np.ndarray), \
        f"Expected smooth_param_final to be an array, got {type(smooth_params_final)}"


def test_ensemble_kalman_smoother_multicam_n_latent():
    """Test ensemble_kalman_smoother_multicam with different n_latent values."""

    keypoint_names = ['kp1', 'kp2']
    data_fields = ['x', 'y', 'likelihood']
    n_cameras = 4
    n_frames = 100
    num_fields = len(data_fields)

    # Create mock MarkerArray with explicit data_fields
    markers_array = np.random.randn(3, n_cameras, n_frames, len(keypoint_names), num_fields)
    markerArray = MarkerArray(markers_array, data_fields=data_fields)  # Ensure data_fields is set

    camera_names = ['cam1', 'cam2']
    quantile_keep_pca = 90
    s_frames = [(0, 10)]

    for n_latent in [2, 3, 5]:  # Test different PCA dimensions
        camera_dfs, _ = ensemble_kalman_smoother_multicam(
            marker_array=markerArray,
            keypoint_names=keypoint_names,
            smooth_param=1,  # Fixed smooth_param to speed up test
            quantile_keep_pca=quantile_keep_pca,
            camera_names=camera_names,
            s_frames=s_frames,
            avg_mode='median',
            inflate_vars=False,
            n_latent=n_latent,  # Testing varying PCA dimensions
        )

        # Ensure the output dataframes exist for each camera
        assert len(camera_dfs) == len(camera_names), \
            f"Expected {len(camera_names)} DataFrames, got {len(camera_dfs)}"


def test_inflate_variance():

    # ------------------------------------------------------------------------
    # test no inflation
    # ------------------------------------------------------------------------
    n_time = 10
    n_cams = 3
    maha_dict = {c: np.ones((n_time, 1)) for c in range(n_cams)}
    v = np.ones((n_time, 2 * n_cams))
    v_new, inflated = inflate_variance(v=v, maha_dict=maha_dict, threshold=5, scalar=2)
    assert not inflated
    assert np.allclose(v, v_new)

    # ------------------------------------------------------------------------
    # test inflation
    # ------------------------------------------------------------------------
    scalar = 2
    v_new, inflated = inflate_variance(v=v, maha_dict=maha_dict, threshold=0.5, scalar=scalar)
    assert inflated
    assert np.allclose(scalar * v, v_new)

    # ------------------------------------------------------------------------
    # test inflation for one view but not the other (>2 cams)
    # ------------------------------------------------------------------------
    n_cams = 3
    maha_dict = {
        0: np.ones((n_time, 1)),
        1: 2 * np.ones((n_time, 1)),
        2: 4 * np.ones((n_time, 1)),
    }
    v = np.ones((n_time, 2 * n_cams))
    scalar = 3
    v_new, inflated = inflate_variance(v=v, maha_dict=maha_dict, threshold=1.5, scalar=scalar)
    assert inflated
    assert np.allclose(v[:, :2], v_new[:, :2])
    assert np.allclose(scalar * v[:, 2:4], v_new[:, 2:4])
    assert np.allclose(scalar * v[:, 4:], v_new[:, 4:])

    # ------------------------------------------------------------------------
    # redo, but with 2 view; variance should be inflated in both
    # ------------------------------------------------------------------------
    n_cams = 2
    maha_dict = {
        0: np.ones((n_time, 1)),
        1: 2 * np.ones((n_time, 1)),
    }
    v = np.ones((n_time, 2 * n_cams))
    scalar = 3
    v_new, inflated = inflate_variance(v=v, maha_dict=maha_dict, threshold=1.5, scalar=scalar)
    assert inflated
    assert np.allclose(scalar * v, v_new)


def test_center_predictions_min_frames():
    """Test that center_predictions correctly handles different initial frame lengths
    post-filtering."""

    # Define dimensions
    n_models, n_cameras, n_frames, n_keypoints = 1, 2, 20, 5  # Example setup
    # n_fields = 5  # (x, y, var_x, var_y, likelihood)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create random data for x, y positions
    data = np.random.randn(n_models, n_cameras, n_frames, n_keypoints, 2) * 10  # (x, y)

    # Create variance data
    variance_data = np.abs(
        np.random.randn(n_models, n_cameras, n_frames, n_keypoints, 2) * 5)  # (var_x, var_y)

    # Set the first frame's variance super high to guarantee filtering
    variance_data[:, :, 0, :, :] = 1e6

    # Randomly select a number of remaining frames to have the same variance value
    shared_variance_value = 2.0  # Example variance threshold
    num_shared_frames = np.random.randint(5, n_frames,
                                          size=n_keypoints)  # Random frames per keypoint
    for k in range(n_keypoints):
        random_indices = np.random.choice(
            np.arange(1, n_frames), num_shared_frames[k], replace=False,
        )
        # Set same variance for these frames
        variance_data[:, :, random_indices, k, :] = shared_variance_value

    # Create random likelihood data (values between 0 and 1)
    likelihood_data = np.random.rand(n_models, n_cameras, n_frames, n_keypoints, 1)

    # Stack all fields into a single array
    full_data = np.concatenate([data, variance_data, likelihood_data], axis=-1)

    # Construct MarkerArray instance with correct fields
    ensemble_marker_array = MarkerArray(
        full_data,
        data_fields=["x", "y", "var_x", "var_y", "likelihood"],
    )

    # Define variance threshold (50th percentile)
    quantile_keep_pca = 50

    # Run the function
    valid_frames_mask, _, emA_good_centered_preds, _ = center_predictions(
        ensemble_marker_array, quantile_keep_pca,
    )

    # Compute expected min_frames by finding the lowest valid frame count across keypoints
    min_frames_expected = min(np.sum(valid_frames_mask, axis=0))  # Count True values per keypoint

    # Ensure the returned good predictions have exactly `min_frames_expected` frames for all kps
    assert emA_good_centered_preds.array.shape[2] == min_frames_expected, \
        f"Expected {min_frames_expected} frames, but got {emA_good_centered_preds.array.shape[2]}"
