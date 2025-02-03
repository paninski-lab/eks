import numpy as np
import pytest

from eks.stats import compute_mahalanobis


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


def test_compute_mahalanobis_singular_matrix():
    """Test compute_mahalanobis behavior with a variance matrix that would be singular."""

    # Create input data
    N = 5  # Number of samples
    C = 2  # Number of views (each view has 2D coordinates)

    np.random.seed(42)
    x = np.random.randn(N, 2 * C)  # Random observations

    # Variance with all zeros (this will cause a singular matrix)
    v = np.zeros((N, 2 * C))

    # 1. Check that function fails with epsilon=0 (singular matrix)
    with pytest.raises(np.linalg.LinAlgError):
        compute_mahalanobis(x, v, epsilon=0, v_quantile_threshold=None)

    # 2. Check that function succeeds with nonzero epsilon
    result = compute_mahalanobis(x, v, epsilon=1e-6, v_quantile_threshold=None)

    # Check that outputs are correctly shaped
    assert 'mahalanobis' in result
    assert 'posterior_variance' in result
    assert 'reconstructed' in result

    for view_idx in range(C):
        assert result['mahalanobis'][view_idx].shape == (N, 1)
        assert result['posterior_variance'][view_idx].shape == (N, 2, 2)

    assert result['reconstructed'].shape == (N, 2 * C)

