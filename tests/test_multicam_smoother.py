import os

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA

from eks.marker_array import MarkerArray
from eks.multicam_smoother import (
    ensemble_kalman_smoother_multicam,
    inflate_variance,
    make_jax_projection_fn,
    make_projection_from_camgroup,
    parse_dist,
    project_3d_covariance_to_2d,
    rodrigues,
    triangulate_3d_models,
)
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
    for k in range(len(keypoint_names)):
        assert smooth_params_final[k] == smooth_param, \
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
    for k in range(len(keypoint_names)):
        assert smooth_params_final[k] == smooth_param, \
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
    for k in range(len(keypoint_names)):
        assert smooth_params_final[k] == smooth_param, \
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
    s_frames = None

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
    s_frames = None

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


# ----------- Calibration Tests ------------
"""
Tests for calibration helpers:
- Rodrigues vs OpenCV
- Distortion parsing semantics
- JAX projection vs cv2.projectPoints (with/without distortion)
- Combined multi-view projector concatenation
- Triangulation wrapper call/shape behavior
- Covariance projection via Jacobian vs finite differences
"""

os.environ.setdefault("JAX_ENABLE_X64", "true")
jax.config.update("jax_enable_x64", True)


def _rng():
    """Deterministic NumPy RNG used across tests for reproducibility."""
    return np.random.default_rng(0)


def _random_rvec_tvec_K_dist(with_dist=True, rng=None):
    """
    Generate a random (rvec, tvec, K, dist) tuple.

    - Distortion uses the standard 5-term model (k1,k2,p1,p2,k3),
      with sensor tilt (tx,ty) set to zero to match the JAX projector (no tilt).
    """
    rng = _rng() if rng is None else rng
    rvec = (rng.normal(size=3) * rng.uniform(0.0, 2.0)).astype(np.float64)
    tvec = (rng.normal(size=3) * 0.5).astype(np.float64)

    fx = rng.uniform(500, 1500)
    fy = rng.uniform(500, 1500)
    cx = rng.uniform(200, 800)
    cy = rng.uniform(200, 800)
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    if with_dist:
        dist = np.zeros(14, dtype=np.float64)
        dist[0] = rng.normal(scale=1e-3)   # k1
        dist[1] = rng.normal(scale=1e-4)   # k2
        dist[2] = rng.normal(scale=1e-4)   # p1
        dist[3] = rng.normal(scale=1e-4)   # p2
        dist[4] = rng.normal(scale=1e-5)   # k3
        dist[12] = 0.0  # tx OFF
        dist[13] = 0.0  # ty OFF
    else:
        dist = np.zeros(14, dtype=np.float64)

    return rvec, tvec, K, dist


def _random_points(N, rng=None):
    """Generate N random 3D points with positive Z (in front of the camera)."""
    rng = _rng() if rng is None else rng
    X = rng.normal(size=(N, 3)).astype(np.float64)
    X[:, 2] = np.abs(X[:, 2]) + 0.5  # in front of camera
    return X


# ---------- _rodrigues ----------
def test_rodrigues_small_angle_matches_cv2():
    """Checks small-angle branch: JAX Rodrigues equals cv2.Rodrigues for ~zero rotation."""
    rvec_small = np.array([1e-10, -2e-10, 3e-10], dtype=np.float64)
    R_cv, _ = cv2.Rodrigues(rvec_small)
    R_jax = rodrigues(jnp.asarray(rvec_small, dtype=jnp.float64))
    np.testing.assert_allclose(np.array(R_jax), R_cv, atol=1e-12, rtol=1e-12)


def test_rodrigues_general_matches_cv2():
    """Checks general-case Rodrigues: random rotations match OpenCV within tight tolerance."""
    rng = _rng()
    for _ in range(5):
        rvec = rng.normal(size=3).astype(np.float64)
        R_cv, _ = cv2.Rodrigues(rvec)
        R_jax = rodrigues(jnp.asarray(rvec, dtype=jnp.float64))
        np.testing.assert_allclose(np.array(R_jax), R_cv, atol=1e-10, rtol=1e-10)


# ---------- _parse_dist ----------
def test_parse_dist_padding_and_ordering():
    """Validates ordering (k1,k2,p1,p2,k3,...) and zero-padding up to 14 OpenCV coeff slots."""
    raw = np.array([0.1, -0.2, 0.01, -0.01, 0.001], dtype=np.float64)
    d = parse_dist(raw)
    for k in ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"]:
        assert k in d
    assert d["k1"] == raw[0]
    assert d["k2"] == raw[1]
    assert d["p1"] == raw[2]
    assert d["p2"] == raw[3]
    assert d["k3"] == raw[4]
    for k in ["k4", "k5", "k6", "s1", "s2", "s3", "s4"]:
        assert d[k] == 0.0


def test_parse_dist_full_length_ignores_tx_ty():
    """Ensures parser drops tx,ty (tilt) and returns only the 12 named distortion fields."""
    raw14 = np.arange(14, dtype=np.float64) / 100.0
    d = parse_dist(raw14)
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty = raw14
    gold = dict(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3, k4=k4, k5=k5, k6=k6, s1=s1, s2=s2, s3=s3, s4=s4)
    for k, v in gold.items():
        assert d[k] == v


# ---------- make_jax_projection_fn vs cv2.projectPoints ----------
def test_project_matches_cv2_no_dist():
    """Checks projection parity with OpenCV when distortion=0."""
    rng = _rng()
    rvec, tvec, K, dist = _random_rvec_tvec_K_dist(with_dist=False, rng=rng)
    proj = make_jax_projection_fn(rvec, tvec, K, dist)
    X = _random_points(100, rng)
    uv_cv, _ = cv2.projectPoints(X, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist)
    uv_cv = uv_cv.reshape(-1, 2)
    uv_jax = np.asarray(proj(jnp.asarray(X)), dtype=np.float64)
    np.testing.assert_allclose(uv_jax, uv_cv, atol=1e-6, rtol=1e-6)


def test_project_matches_cv2_with_dist():
    """Checks projection parity with OpenCV for standard 5-term distortion."""
    rng = _rng()
    rvec, tvec, K, dist = _random_rvec_tvec_K_dist(with_dist=True, rng=rng)
    proj = make_jax_projection_fn(rvec, tvec, K, dist)
    X = _random_points(120, rng)
    uv_cv, _ = cv2.projectPoints(X, rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist)
    uv_cv = uv_cv.reshape(-1, 2)
    uv_jax = np.asarray(proj(jnp.asarray(X)), dtype=np.float64)
    np.testing.assert_allclose(uv_jax, uv_cv, atol=1e-6, rtol=1e-6)


def test_projection_jit_smoke():
    """Smoke test: ensure projector JIT-compiles and returns the expected shape."""
    rng = _rng()
    rvec, tvec, K, dist = _random_rvec_tvec_K_dist(with_dist=True, rng=rng)
    proj = make_jax_projection_fn(rvec, tvec, K, dist)
    proj_jit = jax.jit(proj)
    X = _random_points(32, rng)
    uv = proj_jit(jnp.asarray(X))
    assert np.array(uv).shape == (32, 2)


# ---------- make_projection_from_camgroup ----------
class _MockCam:
    """Minimal camera mock exposing rotation/translation/K/dist getters."""
    def __init__(self, rotation, translation, K, dist):
        self._rotation = rotation
        self._translation = translation
        self._K = K
        self._dist = dist

    def get_rotation(self): return self._rotation
    def get_translation(self): return self._translation
    def get_camera_matrix(self): return self._K
    def get_distortions(self): return self._dist


class _MockCamGroup:
    """Mock camgroup that also provides a dummy triangulate(xy_views) API."""
    def __init__(self, cameras):
        self.cameras = cameras

    def triangulate(self, xy_views, fast=True):
        xy = np.asarray(xy_views)
        return np.array([xy[:, 0].mean(), xy[:, 1].mean(), 1.0], dtype=float)


def test_make_projection_from_camgroup_single_point_concat_order():
    """Verifies combined h_fn concatenates per-camera (2,) projections into (4,) in order."""
    rng = _rng()
    # Camera A given as rotation matrix
    rvecA, tvecA, KA, distA = _random_rvec_tvec_K_dist(with_dist=True, rng=rng)
    RA, _ = cv2.Rodrigues(rvecA)
    camA = _MockCam(RA, tvecA, KA, distA)
    # Camera B given as rvec
    rvecB, tvecB, KB, distB = _random_rvec_tvec_K_dist(with_dist=False, rng=rng)
    camB = _MockCam(rvecB, tvecB, KB, distB)

    cg = _MockCamGroup([camA, camB])
    h_combined, h_cams = make_projection_from_camgroup(cg)

    x = _random_points(1, rng)[0]
    uv0 = np.array(h_cams[0](jnp.asarray(x)))  # (2,)
    uv1 = np.array(h_cams[1](jnp.asarray(x)))  # (2,)
    uv_comb = np.array(h_combined(jnp.asarray(x)))  # expected shape (4,)
    assert uv_comb.shape == (4,), f"Expected (4,), got {uv_comb.shape}"
    np.testing.assert_allclose(uv_comb, np.concatenate([uv0, uv1], axis=0))


# ---------- triangulate_3d_models ----------
class _MockMarkerArray:
    """Minimal stand-in for MarkerArray exposing .shape and .get_array()."""
    def __init__(self, arr):
        self._arr = arr

    @property
    def shape(self):
        return self._arr.shape

    def get_array(self):
        return self._arr


def test_triangulate_3d_models_calls_camgroup_and_shapes():
    """Asserts triangulation wrapper iterates correctly and returns (M,K,T,3) w expected values."""
    rng = _rng()
    M, C, T, K = 2, 3, 5, 4
    arr = rng.normal(size=(M, C, T, K, 3)).astype(np.float64)
    markers = _MockMarkerArray(arr)
    cams = [_MockCam(np.eye(3), np.zeros(3), np.eye(3), np.zeros(14)) for _ in range(C)]
    cg = _MockCamGroup(cams)

    tri = triangulate_3d_models(markers, cg)
    assert tri.shape == (M, K, T, 3)

    for m in range(M):
        for k in range(K):
            for t in range(T):
                xy_views = [arr[m, c, t, k, :2] for c in range(C)]
                expected = np.array([
                    np.mean([xy[0] for xy in xy_views]),
                    np.mean([xy[1] for xy in xy_views]),
                    1.0
                ])
                np.testing.assert_allclose(tri[m, k, t], expected, atol=1e-12)


# ---------- project_3d_covariance_to_2d ----------
def _finite_diff_jacobian(f, x, eps=1e-5):
    """Central-difference Jacobian of a 2D function f: R^3 -> R^2 evaluated at x."""
    x = np.asarray(x, dtype=np.float64)
    J = np.zeros((2, x.size), dtype=np.float64)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        y_plus = np.asarray(f(x + e), dtype=np.float64)
        y_minus = np.asarray(f(x - e), dtype=np.float64)
        J[:, i] = (y_plus - y_minus) / (2.0 * eps)
    return J


def test_project_3d_covariance_to_2d_matches_fd_linearization():
    """
    Validates covariance projection:
    diag(J V J^T) + inflated_vars matches a finite-difference linearization of h_cam at ms_k[t].
    """
    rng = _rng()
    rvec, tvec, K, dist = _random_rvec_tvec_K_dist(with_dist=True, rng=rng)
    h_cam = make_jax_projection_fn(rvec, tvec, K, dist)

    T = 8
    ms_k = _random_points(T, rng)
    A = rng.normal(size=(T, 3, 3)).astype(np.float64) * 1e-2
    Vs_k = np.einsum("tij,tik->tjk", A, A) + np.eye(3)[None] * 1e-6
    inflated = np.abs(rng.normal(size=(T, 3)).astype(np.float64)) * 1e-4

    var_x, var_y = project_3d_covariance_to_2d(ms_k, Vs_k, h_cam, inflated)

    # check a few timesteps
    for t in [0, 3, 5, 7]:
        J_fd = _finite_diff_jacobian(lambda x: np.array(h_cam(jnp.asarray(x))), ms_k[t])
        cov2d_fd = J_fd @ Vs_k[t] @ J_fd.T
        np.testing.assert_allclose(var_x[t] - inflated[t, 0], cov2d_fd[0, 0], rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(var_y[t] - inflated[t, 1], cov2d_fd[1, 1], rtol=1e-4, atol=1e-6)
