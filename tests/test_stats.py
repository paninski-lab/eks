import warnings

import numpy as np
import pytest
from sklearn.decomposition import PCA

from eks.marker_array import MarkerArray
from eks.stats import compute_mahalanobis, compute_pca


def test_compute_pca_basic():
    """Test compute_pca with a simple valid input without a precomputed PCA matrix."""
    n_frames, n_keypoints, n_cameras = 10, 5, 2

    # Create a valid frames mask with True values (all frames are valid for all keypoints)
    valid_frames_mask = np.ones((n_frames, n_keypoints), dtype=bool)

    # Create synthetic MarkerArray data
    emA_centered_preds = MarkerArray(np.random.randn(1, n_cameras, n_frames, n_keypoints, 2))
    emA_good_centered_preds = MarkerArray(np.random.randn(1, n_cameras, n_frames, n_keypoints, 2))

    # Run PCA computation
    ensemble_pca, good_pcs_list = compute_pca(
        valid_frames_mask, emA_centered_preds, emA_good_centered_preds)

    # Assertions
    assert isinstance(ensemble_pca, list)
    assert isinstance(good_pcs_list, list)
    assert len(ensemble_pca) == n_keypoints
    assert len(good_pcs_list) == n_keypoints
    assert all(isinstance(pca, PCA) for pca in ensemble_pca)


def test_compute_pca_with_precomputed_pca():
    """Test compute_pca with a precomputed PCA matrix."""
    n_frames, n_keypoints, n_cameras = 10, 5, 2

    # Create a valid frames mask
    valid_frames_mask = np.ones((n_frames, n_keypoints), dtype=bool)

    # Create synthetic MarkerArray data
    emA_centered_preds = MarkerArray(np.random.randn(1, n_cameras, n_frames, n_keypoints, 2))
    emA_good_centered_preds = MarkerArray(np.random.randn(1, n_cameras, n_frames, n_keypoints, 2))

    # Fit a PCA model for testing
    sample_data = np.random.randn(n_frames, n_cameras * 2)
    precompute_pca = PCA(n_components=3).fit(sample_data)

    # Run PCA computation with precomputed PCA
    ensemble_pca, good_pcs_list = compute_pca(
        valid_frames_mask, emA_centered_preds, emA_good_centered_preds, pca_object=precompute_pca,
    )

    # Assertions
    assert isinstance(ensemble_pca, list)
    assert isinstance(good_pcs_list, list)
    assert len(ensemble_pca) == n_keypoints
    assert len(good_pcs_list) == n_keypoints
    assert all(isinstance(pca, PCA) for pca in ensemble_pca)
    assert all(np.array_equal(pca.components_, precompute_pca.components_) for pca in ensemble_pca)


def test_compute_mahalanobis():

    np.random.seed(0)

    # simple test on output shapes
    n_t = 100
    n_cams = 6
    x = np.random.randn(n_t, 2 * n_cams)
    v = np.ones(x.shape)
    out = compute_mahalanobis(x=x, v=v, n_latent=3, v_quantile_threshold=None)

    assert 'mahalanobis' in out.keys()
    assert len(out['mahalanobis']) == n_cams
    assert out['mahalanobis'][0].shape == (n_t, 1)

    assert 'posterior_variance' in out.keys()
    assert len(out['posterior_variance']) == n_cams
    assert out['posterior_variance'][0].shape == (n_t, 2, 2)

    assert 'reconstructed' in out.keys()
    assert out['reconstructed'].shape == x.shape

    # simple test with likelihoods
    likes = np.random.rand(n_t, n_cams)
    out = compute_mahalanobis(
        x=x, v=v, n_latent=3, likelihoods=likes,
        likelihood_threshold=0.1,
        v_quantile_threshold=None,
    )
    assert out['mahalanobis'][0].shape == (n_t, 1)
    assert out['posterior_variance'][0].shape == (n_t, 2, 2)
    assert out['reconstructed'].shape == x.shape

    # simple test with ensemble variances
    v = 0.01 * np.random.randn(*x.shape) + np.ones(x.shape)
    out = compute_mahalanobis(x=x, v=v, n_latent=3, likelihoods=None, v_quantile_threshold=50)
    assert out['mahalanobis'][0].shape == (n_t, 1)
    assert out['posterior_variance'][0].shape == (n_t, 2, 2)
    assert out['reconstructed'].shape == x.shape

    # test actual data
    n_t = 100
    n_latent = 3
    n_cams = 6
    W = np.random.randn(2 * n_cams, n_latent)
    z = np.random.randn(n_t, n_latent)
    x = z @ W.T
    v = np.ones((n_t, 2 * n_cams))

    # is data reconstructed with the *true* number of latents?
    out = compute_mahalanobis(x, v, n_latent=n_latent, v_quantile_threshold=None)
    assert np.allclose(x, out['reconstructed'])

    # is data *not* reconstructed with the smaller number of latents?
    out = compute_mahalanobis(x, v, n_latent=1, v_quantile_threshold=None)
    assert not np.allclose(x, out['reconstructed'])

    # is the maha smaller/posterior variance larger when observed var is larger?
    s = 10  # scale up ens var
    x2 = np.vstack([x, x])
    v2 = np.vstack([np.ones((n_t, 2 * n_cams)), s * np.ones((n_t, 2 * n_cams))])
    out = compute_mahalanobis(x2, v2, n_latent=n_latent - 1)  # reduce latents -> nonzero errors
    assert np.isclose(out['mahalanobis'][0][0], s * out['mahalanobis'][0][0 + n_t])
    assert np.allclose(s * out['posterior_variance'][0][0], out['posterior_variance'][0][0 + n_t])

    # test when loading matrix/mean are passed in
    compute_mahalanobis(
        x, v, n_latent=n_latent,
        loading_matrix=np.random.randn(2 * n_cams, n_latent),
        mean=np.random.randn(2 * n_cams)
    )


def test_compute_mahalanobis_singular_matrix():
    """Test compute_mahalanobis behavior with a variance matrix that would be singular."""

    # Create input data
    N = 5  # Number of samples
    C = 2  # Number of views (each view has 2D coordinates)

    np.random.seed(42)
    x = np.random.randn(N, 2 * C)  # Random observations

    # Variance with all zeros (this will cause a singular matrix)
    v = np.zeros((N, 2 * C))

    # 1. Check that function issues a warning with epsilon=0 (singular matrix)
    with pytest.warns(RuntimeWarning):
        compute_mahalanobis(x, v, epsilon=0, v_quantile_threshold=None)

    # 2. Check that function succeeds with nonzero epsilon
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Treat warnings as errors to ensure none are raised
        result = compute_mahalanobis(x, v, epsilon=1e-6, v_quantile_threshold=None)

    # Check that outputs are correctly shaped
    assert 'mahalanobis' in result
    assert 'posterior_variance' in result
    assert 'reconstructed' in result

    for view_idx in range(C):
        assert result['mahalanobis'][view_idx].shape == (N, 1)
        assert result['posterior_variance'][view_idx].shape == (N, 2, 2)

    assert result['reconstructed'].shape == (N, 2 * C)
