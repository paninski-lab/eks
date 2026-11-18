import warnings
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_info

from eks.marker_array import MarkerArray
from eks.stats import (
    _fit_factor_analysis,
    _mahalanobis_distance,
    _posterior_latent,
    _posterior_predictive_variance,
    _select_fa_rows,
    compute_mahalanobis,
    compute_pca,
)


class TestComputePca:
    """Test the function compute_pca."""

    def test_compute_pca_basic(self):
        """Test compute_pca with a simple valid input without a precomputed PCA matrix."""
        n_frames, n_keypoints, n_cameras = 10, 5, 2

        # Create a valid frames mask with True values (all frames are valid for all keypoints)
        valid_frames_mask = np.ones((n_frames, n_keypoints), dtype=bool)

        # Create synthetic MarkerArray data
        emA_centered_preds = MarkerArray(np.random.randn(1, n_cameras, n_frames, n_keypoints, 2))
        emA_good_centered_preds = MarkerArray(
            np.random.randn(1, n_cameras, n_frames, n_keypoints, 2),
        )

        # Run PCA computation
        ensemble_pca, good_pcs_list = compute_pca(
            valid_frames_mask, emA_centered_preds, emA_good_centered_preds)

        # Assertions
        assert isinstance(ensemble_pca, list)
        assert isinstance(good_pcs_list, list)
        assert len(ensemble_pca) == n_keypoints
        assert len(good_pcs_list) == n_keypoints
        assert all(isinstance(pca, PCA) for pca in ensemble_pca)

    def test_compute_pca_with_precomputed_pca(self):
        """Test compute_pca with a precomputed PCA matrix."""
        n_frames, n_keypoints, n_cameras = 10, 5, 2

        # Create a valid frames mask
        valid_frames_mask = np.ones((n_frames, n_keypoints), dtype=bool)

        # Create synthetic MarkerArray data
        emA_centered_preds = MarkerArray(np.random.randn(1, n_cameras, n_frames, n_keypoints, 2))
        emA_good_centered_preds = MarkerArray(
            np.random.randn(1, n_cameras, n_frames, n_keypoints, 2),
        )

        # Fit a PCA model for testing
        sample_data = np.random.randn(n_frames, n_cameras * 2)
        precompute_pca = PCA(n_components=3).fit(sample_data)

        # Run PCA computation with precomputed PCA
        ensemble_pca, good_pcs_list = compute_pca(
            valid_frames_mask, emA_centered_preds, emA_good_centered_preds,
            pca_object=precompute_pca,
        )

        # Assertions
        assert isinstance(ensemble_pca, list)
        assert isinstance(good_pcs_list, list)
        assert len(ensemble_pca) == n_keypoints
        assert len(good_pcs_list) == n_keypoints
        assert all(isinstance(pca, PCA) for pca in ensemble_pca)
        assert all(
            np.array_equal(pca.components_, precompute_pca.components_)  # type: ignore[arg-type]
            for pca in ensemble_pca
        )


# ---------------------------------------------------------------------------------------------
# reference implementations
# naive per-frame versions of the posterior computations; deliberately independent of the code
# under test so that vectorized implementations can be checked against them
# ---------------------------------------------------------------------------------------------


def _ref_posterior_latent(x, v, W, mu, epsilon):
    """Per-frame posterior mean and covariance of latents."""
    n, n_latent = v.shape[0], W.shape[1]
    z_hat = np.zeros((n, n_latent))
    B = np.zeros((n, n_latent, n_latent))
    for i in range(n):
        precision = W.T @ np.diag(1.0 / (v[i] + epsilon)) @ W
        B[i] = np.linalg.inv(precision)
        z_hat[i] = B[i] @ W.T @ np.diag(1.0 / (v[i] + epsilon)) @ (x[i] - mu)
    return z_hat, B


def _ref_posterior_predictive_variance(v, W, B):
    """Per-frame, per-view posterior predictive covariance."""
    n, n_views = v.shape[0], v.shape[1] // 2
    Q = {c: np.zeros((n, 2, 2)) for c in range(n_views)}
    for i in range(n):
        for c in range(n_views):
            sl = slice(2 * c, 2 * c + 2)
            Q[c][i] = np.diag(v[i, sl]) + W[sl] @ B[i] @ W[sl].T
    return Q


def _ref_mahalanobis_distance(diff, Q):
    """Per-frame, per-view squared Mahalanobis distance."""
    n, n_views = diff.shape[0], diff.shape[1] // 2
    M = {c: np.zeros((n, 1)) for c in range(n_views)}
    for i in range(n):
        for c in range(n_views):
            sl = slice(2 * c, 2 * c + 2)
            M[c][i, 0] = diff[i, sl] @ np.linalg.solve(Q[c][i], diff[i, sl])
    return M


def _make_inputs(n=40, n_views=3, n_latent=2, seed=0, noise=0.5):
    """Generate observations, heteroscedastic variances, and a loading matrix / mean."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(2 * n_views, n_latent))
    mu = rng.normal(size=2 * n_views)
    z = rng.normal(size=(n, n_latent))
    x = z @ W.T + mu + noise * rng.normal(size=(n, 2 * n_views))
    v = rng.uniform(0.2, 3.0, size=(n, 2 * n_views))
    return x, v, W, mu, z


def _random_spd(rng, n, dim=2):
    """Generate a batch of random symmetric positive definite matrices."""
    A = rng.normal(size=(n, dim, dim))
    return A @ np.swapaxes(A, 1, 2) + 0.5 * np.eye(dim)


class TestSelectFaRows:
    """Test the function _select_fa_rows."""

    def test_select_fa_rows_no_filters_keeps_all(self):
        v = np.random.default_rng(0).uniform(1, 2, size=(20, 6))
        mask = _select_fa_rows(v, None)
        assert mask.dtype == bool
        assert mask.shape == (20,)
        assert mask.all()

    def test_select_fa_rows_variance_quantile_filter(self):
        # max variance per row is 1..10
        v = np.repeat(np.arange(1.0, 11.0)[:, None], 4, axis=1)
        mask = _select_fa_rows(v, 50.0)
        # strictly below the 50th percentile (5.5) -> rows with max var 1..5
        assert mask.sum() == 5
        assert mask[:5].all() and not mask[5:].any()

    def test_select_fa_rows_variance_uses_max_across_views(self):
        v = np.ones((10, 4))
        v[:, 3] = np.arange(10.0)  # only one column varies; it determines the row max
        mask = _select_fa_rows(v, 50.0)
        assert mask.sum() == 5
        assert mask[:5].all()

    def test_select_fa_rows_constant_variance_keeps_no_rows(self):
        # strict inequality against the percentile -> nothing survives; callers must avoid this
        mask = _select_fa_rows(np.ones((10, 4)), 50.0)
        assert not mask.any()

    def test_select_fa_rows_quantile_100_keeps_all_but_max(self):
        v = np.repeat(np.arange(1.0, 11.0)[:, None], 4, axis=1)
        mask = _select_fa_rows(v, 100.0)
        assert mask.sum() == 9
        assert not mask[-1]


class TestFitFactorAnalysis:
    """Test the function _fit_factor_analysis."""

    def test_fit_factor_analysis_output_shapes(self):
        x, _, _, _, _ = _make_inputs(n=60, n_views=3, n_latent=2)
        W, mu = _fit_factor_analysis(x, n_latent=2)
        assert W.shape == (6, 2)
        assert mu.shape == (6,)

    def test_fit_factor_analysis_mean_matches_data_mean(self):
        x, _, _, _, _ = _make_inputs(n=60)
        _, mu = _fit_factor_analysis(x, n_latent=2)
        assert np.allclose(mu, x.mean(axis=0))

    def test_fit_factor_analysis_recovers_subspace(self):
        x, _, W_true, _, _ = _make_inputs(n=500, noise=0.01)
        W, _ = _fit_factor_analysis(x, n_latent=2)
        # the column spaces should coincide: projecting W_true onto span(W) is lossless
        proj = W @ np.linalg.pinv(W) @ W_true
        assert np.allclose(proj, W_true, atol=0.1)

    def test_fit_factor_analysis_max_iters_is_forwarded(self):
        x, _, _, _, _ = _make_inputs(n=60)
        with patch('eks.stats.FactorAnalysis', wraps=FactorAnalysis) as mock_fa:
            _fit_factor_analysis(x, n_latent=2, max_iters=123)
        assert mock_fa.call_args.kwargs['max_iter'] == 123

    def test_fit_factor_analysis_default_max_iters_is_500(self):
        x, _, _, _, _ = _make_inputs(n=60)
        with patch('eks.stats.FactorAnalysis', wraps=FactorAnalysis) as mock_fa:
            _fit_factor_analysis(x, n_latent=2)
        assert mock_fa.call_args.kwargs['max_iter'] == 500

    def test_fit_factor_analysis_warns_when_iteration_cap_is_hit(self):
        x, _, _, _, _ = _make_inputs(n=60)
        with pytest.warns(ConvergenceWarning):
            _fit_factor_analysis(x, n_latent=2, max_iters=1)

    def test_fit_factor_analysis_runs_with_single_threaded_blas(self):
        x, _, _, _, _ = _make_inputs(n=60)
        seen = []
        original_fit = FactorAnalysis.fit

        def spy_fit(self, X, y=None):
            seen.append({info['num_threads'] for info in threadpool_info()})
            return original_fit(self, X, y)

        with patch.object(FactorAnalysis, 'fit', spy_fit):
            _fit_factor_analysis(x, n_latent=2)
        assert seen == [{1}]

    def test_fit_factor_analysis_restores_blas_thread_limits(self):
        x, _, _, _, _ = _make_inputs(n=60)
        before = [(i['internal_api'], i['num_threads']) for i in threadpool_info()]
        _fit_factor_analysis(x, n_latent=2)
        after = [(i['internal_api'], i['num_threads']) for i in threadpool_info()]
        assert before == after

    def test_fit_factor_analysis_deterministic(self):
        x, _, _, _, _ = _make_inputs(n=60)
        W1, mu1 = _fit_factor_analysis(x, n_latent=2)
        W2, mu2 = _fit_factor_analysis(x, n_latent=2)
        assert np.allclose(W1, W2)
        assert np.allclose(mu1, mu2)


class TestPosteriorLatent:
    """Test the function _posterior_latent."""

    def test_posterior_latent_shapes(self):
        x, v, W, mu, _ = _make_inputs(n=25, n_views=3, n_latent=2)
        z_hat, B = _posterior_latent(x, v, W, mu, 1e-6)
        assert z_hat.shape == (25, 2)
        assert B.shape == (25, 2, 2)

    def test_posterior_latent_matches_reference(self):
        x, v, W, mu, _ = _make_inputs(n=40, n_views=4, n_latent=3, seed=1)
        z_hat, B = _posterior_latent(x, v, W, mu, 1e-6)
        z_ref, B_ref = _ref_posterior_latent(x, v, W, mu, 1e-6)
        assert np.allclose(z_hat, z_ref)
        assert np.allclose(B, B_ref)

    def test_posterior_latent_covariance_is_inverse_precision(self):
        x, v, W, mu, _ = _make_inputs(n=10)
        eps = 1e-3
        _, B = _posterior_latent(x, v, W, mu, eps)
        for i in range(10):
            precision = W.T @ np.diag(1.0 / (v[i] + eps)) @ W
            assert np.allclose(np.linalg.inv(B[i]), precision)

    def test_posterior_latent_covariance_symmetric_positive_definite(self):
        x, v, W, mu, _ = _make_inputs(n=30, n_latent=3, n_views=4)
        _, B = _posterior_latent(x, v, W, mu, 1e-6)
        assert np.allclose(B, np.swapaxes(B, 1, 2))
        assert (np.linalg.eigvalsh(B) > 0).all()

    def test_posterior_latent_recovers_noise_free_latents(self):
        x, v, W, mu, z = _make_inputs(n=20, noise=0.0)
        z_hat, _ = _posterior_latent(x, v, W, mu, 1e-6)
        assert np.allclose(z_hat, z, atol=1e-6)

    def test_posterior_latent_rows_are_independent(self):
        x, v, W, mu, _ = _make_inputs(n=30)
        perm = np.random.default_rng(3).permutation(30)
        z_hat, B = _posterior_latent(x, v, W, mu, 1e-6)
        z_perm, B_perm = _posterior_latent(x[perm], v[perm], W, mu, 1e-6)
        assert np.allclose(z_hat[perm], z_perm)
        assert np.allclose(B[perm], B_perm)

    def test_posterior_latent_single_row(self):
        x, v, W, mu, _ = _make_inputs(n=1)
        z_hat, B = _posterior_latent(x, v, W, mu, 1e-6)
        assert z_hat.shape == (1, 2)
        assert B.shape == (1, 2, 2)

    def test_posterior_latent_single_latent(self):
        x, v, W, mu, _ = _make_inputs(n=15, n_latent=1)
        z_hat, B = _posterior_latent(x, v, W, mu, 1e-6)
        z_ref, B_ref = _ref_posterior_latent(x, v, W, mu, 1e-6)
        assert z_hat.shape == (15, 1)
        assert np.allclose(z_hat, z_ref)
        assert np.allclose(B, B_ref)

    def test_posterior_latent_epsilon_regularizes_zero_variance(self):
        x, _, W, mu, _ = _make_inputs(n=5)
        v = np.zeros_like(x)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            z_hat, B = _posterior_latent(x, v, W, mu, 1e-6)
        assert np.isfinite(z_hat).all()
        assert np.isfinite(B).all()

    def test_posterior_latent_covariance_scales_with_variance(self):
        # scaling all observation variances scales the posterior covariance by the same factor
        x, v, W, mu, _ = _make_inputs(n=5)
        _, B1 = _posterior_latent(x, v, W, mu, 0.0)
        _, B10 = _posterior_latent(x, 10 * v, W, mu, 0.0)
        assert np.allclose(B10, 10 * B1)


class TestPosteriorPredictiveVariance:
    """Test the function _posterior_predictive_variance."""

    def test_posterior_predictive_variance_shapes_and_keys(self):
        _, v, W, _, _ = _make_inputs(n=12, n_views=4)
        B = _random_spd(np.random.default_rng(0), 12, dim=2)
        Q = _posterior_predictive_variance(v, W, B)
        assert sorted(Q.keys()) == [0, 1, 2, 3]
        assert all(q.shape == (12, 2, 2) for q in Q.values())

    def test_posterior_predictive_variance_matches_reference(self):
        x, v, W, mu, _ = _make_inputs(n=30, n_views=4, n_latent=3, seed=2)
        _, B = _ref_posterior_latent(x, v, W, mu, 1e-6)
        Q = _posterior_predictive_variance(v, W, B)
        Q_ref = _ref_posterior_predictive_variance(v, W, B)
        for c in Q_ref:
            assert np.allclose(Q[c], Q_ref[c])

    def test_posterior_predictive_variance_symmetric_positive_definite(self):
        x, v, W, mu, _ = _make_inputs(n=30)
        _, B = _ref_posterior_latent(x, v, W, mu, 1e-6)
        Q = _posterior_predictive_variance(v, W, B)
        for q in Q.values():
            assert np.allclose(q, np.swapaxes(q, 1, 2))
            assert (np.linalg.eigvalsh(q) > 0).all()

    def test_posterior_predictive_variance_zero_latent_covariance_gives_observation_noise(self):
        _, v, W, _, _ = _make_inputs(n=8, n_views=3)
        Q = _posterior_predictive_variance(v, W, np.zeros((8, 2, 2)))
        for c, q in Q.items():
            expected = np.stack([np.diag(v[i, 2 * c:2 * c + 2]) for i in range(8)])
            assert np.allclose(q, expected)

    def test_posterior_predictive_variance_dominates_observation_noise(self):
        x, v, W, mu, _ = _make_inputs(n=20)
        _, B = _ref_posterior_latent(x, v, W, mu, 1e-6)
        Q = _posterior_predictive_variance(v, W, B)
        for c, q in Q.items():
            extra = q - np.stack([np.diag(v[i, 2 * c:2 * c + 2]) for i in range(20)])
            assert (np.linalg.eigvalsh(extra) > -1e-10).all()

    def test_posterior_predictive_variance_views_use_own_columns(self):
        # changing one view's variance must not alter another view's observation-noise term
        _, v, W, _, _ = _make_inputs(n=6, n_views=3)
        B = np.zeros((6, 2, 2))
        v2 = v.copy()
        v2[:, 0:2] *= 5
        Q = _posterior_predictive_variance(v, W, B)
        Q2 = _posterior_predictive_variance(v2, W, B)
        assert not np.allclose(Q[0], Q2[0])
        assert np.allclose(Q[1], Q2[1])
        assert np.allclose(Q[2], Q2[2])

    def test_posterior_predictive_variance_single_row(self):
        _, v, W, _, _ = _make_inputs(n=1)
        B = _random_spd(np.random.default_rng(0), 1, dim=2)
        Q = _posterior_predictive_variance(v, W, B)
        assert all(q.shape == (1, 2, 2) for q in Q.values())


class TestMahalanobisDistance:
    """Test the function _mahalanobis_distance."""

    def test_mahalanobis_distance_shapes_and_keys(self):
        rng = np.random.default_rng(0)
        diff = rng.normal(size=(9, 6))
        Q = {c: _random_spd(rng, 9) for c in range(3)}
        M = _mahalanobis_distance(diff, Q)
        assert sorted(M.keys()) == [0, 1, 2]
        assert all(m.shape == (9, 1) for m in M.values())

    def test_mahalanobis_distance_matches_reference(self):
        rng = np.random.default_rng(1)
        diff = rng.normal(size=(25, 8))
        Q = {c: _random_spd(rng, 25) for c in range(4)}
        M = _mahalanobis_distance(diff, Q)
        M_ref = _ref_mahalanobis_distance(diff, Q)
        for c in M_ref:
            assert np.allclose(M[c], M_ref[c])

    def test_mahalanobis_distance_identity_covariance_is_squared_norm(self):
        rng = np.random.default_rng(2)
        diff = rng.normal(size=(10, 4))
        Q = {c: np.tile(np.eye(2), (10, 1, 1)) for c in range(2)}
        M = _mahalanobis_distance(diff, Q)
        for c in range(2):
            assert np.allclose(M[c][:, 0], (diff[:, 2 * c:2 * c + 2] ** 2).sum(axis=1))

    def test_mahalanobis_distance_zero_residual_is_zero(self):
        rng = np.random.default_rng(3)
        Q = {c: _random_spd(rng, 5) for c in range(2)}
        M = _mahalanobis_distance(np.zeros((5, 4)), Q)
        assert all(np.allclose(m, 0) for m in M.values())

    def test_mahalanobis_distance_nonnegative(self):
        rng = np.random.default_rng(4)
        diff = rng.normal(size=(50, 6))
        Q = {c: _random_spd(rng, 50) for c in range(3)}
        M = _mahalanobis_distance(diff, Q)
        assert all((m >= 0).all() for m in M.values())

    def test_mahalanobis_distance_scales_inversely_with_covariance(self):
        rng = np.random.default_rng(5)
        diff = rng.normal(size=(7, 4))
        Q = {c: _random_spd(rng, 7) for c in range(2)}
        Q_big = {c: 4.0 * q for c, q in Q.items()}
        M = _mahalanobis_distance(diff, Q)
        M_big = _mahalanobis_distance(diff, Q_big)
        for c in M:
            assert np.allclose(M[c], 4.0 * M_big[c])

    def test_mahalanobis_distance_non_diagonal_covariance(self):
        # correlated covariance: distance along the correlated direction is smaller
        Q = {0: np.array([[[1.0, 0.9], [0.9, 1.0]]])}
        along = _mahalanobis_distance(np.array([[1.0, 1.0]]), Q)[0][0, 0]
        across = _mahalanobis_distance(np.array([[1.0, -1.0]]), Q)[0][0, 0]
        assert np.isclose(along, 2 / 1.9)
        assert np.isclose(across, 2 / 0.1)

    def test_mahalanobis_distance_single_row(self):
        rng = np.random.default_rng(6)
        Q = {c: _random_spd(rng, 1) for c in range(2)}
        M = _mahalanobis_distance(rng.normal(size=(1, 4)), Q)
        assert all(m.shape == (1, 1) for m in M.values())


class TestComputeMahalanobis:
    """Test the function compute_mahalanobis."""

    def test_compute_mahalanobis_output_structure(self):
        np.random.seed(0)
        n_t, n_cams = 100, 6
        x = np.random.randn(n_t, 2 * n_cams)
        v = np.ones(x.shape)
        out = compute_mahalanobis(x=x, v=v, n_latent=3, v_quantile_threshold=None)

        assert set(out.keys()) == {'mahalanobis', 'posterior_variance', 'reconstructed'}
        assert len(out['mahalanobis']) == n_cams
        assert out['mahalanobis'][0].shape == (n_t, 1)
        assert len(out['posterior_variance']) == n_cams
        assert out['posterior_variance'][0].shape == (n_t, 2, 2)
        assert out['reconstructed'].shape == x.shape

    def test_compute_mahalanobis_with_ensemble_variances(self):
        np.random.seed(0)
        n_t, n_cams = 100, 6
        x = np.random.randn(n_t, 2 * n_cams)
        v = 0.01 * np.random.randn(*x.shape) + np.ones(x.shape)
        out = compute_mahalanobis(
            x=x, v=v, n_latent=3, v_quantile_threshold=50,
        )
        assert out['mahalanobis'][0].shape == (n_t, 1)
        assert out['posterior_variance'][0].shape == (n_t, 2, 2)
        assert out['reconstructed'].shape == x.shape

    def test_compute_mahalanobis_reconstructs_with_true_number_of_latents(self):
        np.random.seed(0)
        n_t, n_latent, n_cams = 100, 3, 6
        W = np.random.randn(2 * n_cams, n_latent)
        x = np.random.randn(n_t, n_latent) @ W.T
        v = np.ones((n_t, 2 * n_cams))
        out = compute_mahalanobis(x, v, n_latent=n_latent, v_quantile_threshold=None)
        assert np.allclose(x, out['reconstructed'])

    def test_compute_mahalanobis_does_not_reconstruct_with_too_few_latents(self):
        np.random.seed(0)
        n_t, n_latent, n_cams = 100, 3, 6
        W = np.random.randn(2 * n_cams, n_latent)
        x = np.random.randn(n_t, n_latent) @ W.T
        v = np.ones((n_t, 2 * n_cams))
        out = compute_mahalanobis(x, v, n_latent=1, v_quantile_threshold=None)
        assert not np.allclose(x, out['reconstructed'])

    def test_compute_mahalanobis_scales_with_variance(self):
        # larger observed variance -> proportionally smaller maha, larger posterior variance
        np.random.seed(0)
        n_t, n_latent, n_cams, s = 100, 3, 6, 10
        W = np.random.randn(2 * n_cams, n_latent)
        x = np.random.randn(n_t, n_latent) @ W.T
        x2 = np.vstack([x, x])
        v2 = np.vstack([np.ones((n_t, 2 * n_cams)), s * np.ones((n_t, 2 * n_cams))])
        out = compute_mahalanobis(x2, v2, n_latent=n_latent - 1)  # fewer latents -> nonzero errors
        assert np.isclose(out['mahalanobis'][0][0], s * out['mahalanobis'][0][n_t])
        assert np.allclose(s * out['posterior_variance'][0][0], out['posterior_variance'][0][n_t])

    def test_compute_mahalanobis_with_loading_matrix_and_mean_provided(self):
        np.random.seed(0)
        n_t, n_latent, n_cams = 100, 3, 6
        x = np.random.randn(n_t, 2 * n_cams)
        v = np.ones((n_t, 2 * n_cams))
        out = compute_mahalanobis(
            x, v, n_latent=n_latent,
            loading_matrix=np.random.randn(2 * n_cams, n_latent),
            mean=np.random.randn(2 * n_cams),
        )
        assert out['reconstructed'].shape == x.shape

    def test_compute_mahalanobis_skips_factor_analysis_when_parameters_provided(self):
        x, v, W, mu, _ = _make_inputs()
        with patch('eks.stats._fit_factor_analysis') as mock_fit:
            compute_mahalanobis(x, v, n_latent=2, loading_matrix=W, mean=mu)
        mock_fit.assert_not_called()

    def test_compute_mahalanobis_fits_factor_analysis_on_selected_rows_only(self):
        x, v, _, _, _ = _make_inputs(n=40)
        expected_rows = _select_fa_rows(v, 50.0)
        W, mu = _fit_factor_analysis(x[expected_rows], 2)
        with patch('eks.stats._fit_factor_analysis', return_value=(W, mu)) as mock_fit:
            compute_mahalanobis(x, v, n_latent=2, v_quantile_threshold=50.0)
        fit_x, fit_n_latent, fit_max_iters = mock_fit.call_args.args
        assert np.array_equal(fit_x, x[expected_rows])
        assert fit_n_latent == 2
        assert fit_max_iters == 500  # default

    def test_compute_mahalanobis_forwards_max_iters_fa(self):
        x, v, _, _, _ = _make_inputs(n=40)
        W, mu = _fit_factor_analysis(x, 2)
        with patch('eks.stats._fit_factor_analysis', return_value=(W, mu)) as mock_fit:
            compute_mahalanobis(x, v, n_latent=2, v_quantile_threshold=None, max_iters_fa=7)
        assert mock_fit.call_args.args[2] == 7

    def test_compute_mahalanobis_matches_reference_with_fixed_parameters(self):
        # end-to-end check against naive per-frame implementations
        x, v, W, mu, _ = _make_inputs(n=60, n_views=4, n_latent=3, seed=7)
        eps = 1e-6
        out = compute_mahalanobis(
            x, v, n_latent=3, loading_matrix=W, mean=mu, epsilon=eps,
        )
        z_ref, B_ref = _ref_posterior_latent(x, v, W, mu, eps)
        xhat_ref = z_ref @ W.T + mu
        Q_ref = _ref_posterior_predictive_variance(v, W, B_ref)
        M_ref = _ref_mahalanobis_distance(x - xhat_ref, Q_ref)
        assert np.allclose(out['reconstructed'], xhat_ref)
        for c in range(4):
            assert np.allclose(out['posterior_variance'][c], Q_ref[c])
            assert np.allclose(out['mahalanobis'][c], M_ref[c])

    def test_compute_mahalanobis_fitted_path_matches_fixed_parameters_path(self):
        # passing the FA solution back in must reproduce the fitted-path output exactly
        x, v, _, _, _ = _make_inputs(n=60, seed=8)
        out_fit = compute_mahalanobis(x, v, n_latent=2, v_quantile_threshold=None)
        W, mu = _fit_factor_analysis(x, 2)
        out_fixed = compute_mahalanobis(x, v, n_latent=2, loading_matrix=W, mean=mu)
        assert np.allclose(out_fit['reconstructed'], out_fixed['reconstructed'])
        for c in range(3):
            assert np.allclose(out_fit['mahalanobis'][c], out_fixed['mahalanobis'][c])
            assert np.allclose(
                out_fit['posterior_variance'][c], out_fixed['posterior_variance'][c],
            )

    def test_compute_mahalanobis_flags_outlier_view(self):
        # corrupting one view of one frame makes that view's distance dominate
        x, v, W, mu, _ = _make_inputs(n=50, n_views=4, noise=0.05, seed=9)
        x_bad = x.copy()
        x_bad[10, 2:4] += 50.0  # view 1
        out = compute_mahalanobis(x_bad, v, n_latent=2, loading_matrix=W, mean=mu)
        assert out['mahalanobis'][1][10, 0] > 100
        assert out['mahalanobis'][1][10, 0] > 10 * np.median(out['mahalanobis'][1])

    def test_compute_mahalanobis_does_not_modify_inputs(self):
        x, v, W, mu, _ = _make_inputs()
        x0, v0 = x.copy(), v.copy()
        compute_mahalanobis(x, v, n_latent=2)
        assert np.array_equal(x, x0)
        assert np.array_equal(v, v0)

    def test_compute_mahalanobis_rows_are_independent_given_parameters(self):
        x, v, W, mu, _ = _make_inputs(n=30)
        perm = np.random.default_rng(0).permutation(30)
        out = compute_mahalanobis(x, v, n_latent=2, loading_matrix=W, mean=mu)
        out_perm = compute_mahalanobis(x[perm], v[perm], n_latent=2, loading_matrix=W, mean=mu)
        assert np.allclose(out['reconstructed'][perm], out_perm['reconstructed'])
        for c in range(3):
            assert np.allclose(out['mahalanobis'][c][perm], out_perm['mahalanobis'][c])

    def test_compute_mahalanobis_two_views(self):
        x, v, W, mu, _ = _make_inputs(n=30, n_views=2, n_latent=1)
        out = compute_mahalanobis(x, v, n_latent=1, loading_matrix=W, mean=mu)
        assert sorted(out['mahalanobis'].keys()) == [0, 1]

    def test_compute_mahalanobis_singular_variance(self):
        np.random.seed(42)
        n, n_views = 5, 2
        x = np.random.randn(n, 2 * n_views)
        v = np.zeros((n, 2 * n_views))  # all-zero variance -> singular without epsilon

        # epsilon=0 produces a warning
        with pytest.warns(RuntimeWarning):
            compute_mahalanobis(x, v, epsilon=0, v_quantile_threshold=None)

        # nonzero epsilon succeeds without warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            result = compute_mahalanobis(x, v, epsilon=1e-6, v_quantile_threshold=None)

        assert set(result.keys()) == {'mahalanobis', 'posterior_variance', 'reconstructed'}
        for view_idx in range(n_views):
            assert result['mahalanobis'][view_idx].shape == (n, 1)
            assert result['posterior_variance'][view_idx].shape == (n, 2, 2)
        assert result['reconstructed'].shape == (n, 2 * n_views)
