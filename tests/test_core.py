import pytest
import numpy as np
import jax.numpy as jnp
import jax
import pandas as pd
from eks.core import ensemble, kalman_dot, forward_pass, backward_pass, compute_nll, jax_ensemble
from collections import defaultdict


def test_ensemble():
    # Simulate marker data with three models, each with two keypoints and 5 samples
    np.random.seed(0)
    num_samples = 5
    num_keypoints = 2
    markers_list = []
    keys = ['keypoint_1', 'keypoint_2']

    # Create random data for three different marker DataFrames
    # Adjust column names to match the function's expected 'keypoint_likelihood' format
    for i in range(3):
        data = {
            'keypoint_1': np.random.rand(num_samples),
            'keypoint_likelihood': np.random.rand(num_samples),  # Expected naming format
            'keypoint_2': np.random.rand(num_samples),
            'keypoint_likelihod': np.random.rand(num_samples)    # Expected naming format
        }
        markers_list.append(pd.DataFrame(data))

    # Run the ensemble function with 'median' mode
    ensemble_preds, ensemble_vars, ensemble_stacks, keypoints_avg_dict, \
    keypoints_var_dict, keypoints_stack_dict = ensemble(markers_list, keys, mode='median')

    # Verify shapes of output arrays
    assert ensemble_preds.shape == (num_samples, num_keypoints), \
        f"Expected shape {(num_samples, num_keypoints)}, got {ensemble_preds.shape}"
    assert ensemble_vars.shape == (num_samples, num_keypoints), \
        f"Expected shape {(num_samples, num_keypoints)}, got {ensemble_vars.shape}"
    assert ensemble_stacks.shape == (3, num_samples, num_keypoints), \
        f"Expected shape {(3, num_samples, num_keypoints)}, got {ensemble_stacks.shape}"

    # Verify contents of dictionaries
    assert set(keypoints_avg_dict.keys()) == set(keys), \
        f"Expected keys {keys}, got {keypoints_avg_dict.keys()}"
    assert set(keypoints_var_dict.keys()) == set(keys), \
        f"Expected keys {keys}, got {keypoints_var_dict.keys()}"
    assert len(keypoints_stack_dict) == 3, \
        f"Expected 3 models, got {len(keypoints_stack_dict)}"

    # Check values for a keypoint (manually compute median and variance)
    for key in keys:
        stack = np.array([df[key].values for df in markers_list]).T
        expected_median = np.nanmedian(stack, axis=1)
        expected_variance = np.nanvar(stack, axis=1)

        assert np.allclose(keypoints_avg_dict[key], expected_median), \
            f"Expected {expected_median} for {key}, got {keypoints_avg_dict[key]}"
        assert np.allclose(keypoints_var_dict[key], expected_variance), \
            f"Expected {expected_variance} for {key}, got {keypoints_var_dict[key]}"

    # Run the ensemble function with 'confidence_weighted_mean' mode
    ensemble_preds, ensemble_vars, ensemble_stacks, keypoints_avg_dict, \
    keypoints_var_dict, keypoints_stack_dict = ensemble(markers_list, keys,
                                                        mode='confidence_weighted_mean')

    # Verify shapes of output arrays again
    assert ensemble_preds.shape == (num_samples, num_keypoints), \
        f"Expected shape {(num_samples, num_keypoints)}, got {ensemble_preds.shape}"
    assert ensemble_vars.shape == (num_samples, num_keypoints), \
        f"Expected shape {(num_samples, num_keypoints)}, got {ensemble_vars.shape}"
    assert ensemble_stacks.shape == (3, num_samples, num_keypoints), \
        f"Expected shape {(3, num_samples, num_keypoints)}, got {ensemble_stacks.shape}"

    # Verify likelihood-based weighted averaging calculations
    for key in keys:
        stack = np.array([df[key].values for df in markers_list]).T
        likelihood_stack = np.array([df[key[:-1] + 'likelihood'].values for df in markers_list]).T
        conf_per_keypoint = np.sum(likelihood_stack, axis=1)
        weighted_mean = np.sum(stack * likelihood_stack, axis=1) / conf_per_keypoint
        expected_variance = np.nanvar(stack, axis=1) / (
                np.sum(likelihood_stack, axis=1) / likelihood_stack.shape[1])

        assert np.allclose(keypoints_avg_dict[key], weighted_mean), \
            f"Expected {weighted_mean} for {key}, got {keypoints_avg_dict[key]}"
        assert np.allclose(keypoints_var_dict[key], expected_variance), \
            f"Expected {expected_variance} for {key}, got {keypoints_var_dict[key]}"


def test_kalman_dot_basic():
    # Basic test with random matrices
    n_keypoints = 5
    n_latents = 3

    innovation = np.random.randn(n_keypoints)
    V = np.eye(n_latents)
    C = np.random.randn(n_keypoints, n_latents)
    R = np.eye(n_keypoints)

    # Run kalman_dot
    Ks, innovation_cov = kalman_dot(innovation, V, C, R)

    # Check output shapes
    assert Ks.shape == (n_latents,), f"Expected shape {(n_latents,)}, got {Ks.shape}"
    assert innovation_cov.shape == (n_keypoints, n_keypoints), \
        f"Expected shape {(n_keypoints, n_keypoints)}, got {innovation_cov.shape}"

    # Ensure that innovation_cov is symmetric and positive semi-definite
    assert np.allclose(innovation_cov, innovation_cov.T), "Expected innovation_cov to be symmetric"
    eigvals = np.linalg.eigvalsh(innovation_cov)
    assert np.all(eigvals >= 0), "Expected innovation_cov to be positive semi-definite"

    # Check that Ks and innovation_cov have finite values
    assert np.isfinite(Ks).all(), "Expected finite values in Ks"
    assert np.isfinite(innovation_cov).all(), "Expected finite values in innovation_cov"


def test_kalman_dot_zero_matrices():
    # Test with zero matrices for stability
    n_keypoints = 4
    n_latents = 2

    innovation = np.zeros(n_keypoints)
    V = np.zeros((n_latents, n_latents))
    C = np.zeros((n_keypoints, n_latents))
    R = np.zeros((n_keypoints, n_keypoints))

    # Add a small regularization term to R to avoid singularity
    epsilon = 1e-6
    R += epsilon * np.eye(n_keypoints)

    # Run kalman_dot
    Ks, innovation_cov = kalman_dot(innovation, V, C, R)

    # Verify that the output shapes are as expected
    assert Ks.shape == (n_latents,), f"Expected shape {(n_latents,)}, got {Ks.shape}"
    assert innovation_cov.shape == (n_keypoints, n_keypoints), \
        f"Expected shape {(n_keypoints, n_keypoints)}, got {innovation_cov.shape}"

    # Check that innovation_cov has finite values and is symmetric
    assert np.isfinite(innovation_cov).all(), "Expected finite values in innovation_cov"
    assert np.allclose(innovation_cov, innovation_cov.T), "Expected innovation_cov to be symmetric"

    # Check that Ks has finite values
    assert np.isfinite(Ks).all(), "Expected finite values in Ks"


def test_kalman_dot_singular_innovation_cov():
    # Test for singular innovation_cov by making R and C*V*C.T equal
    n_keypoints = 3
    n_latents = 2

    innovation = np.random.randn(n_keypoints)
    V = np.eye(n_latents)
    C = np.ones((n_keypoints, n_latents))  # Constant values lead to rank-deficient product
    R = -np.dot(C, np.dot(V, C.T))  # Makes innovation_cov close to zero matrix

    # Add a small regularization term to R to avoid singularity
    epsilon = 1e-6
    R += epsilon * np.eye(n_keypoints)

    # Run kalman_dot and check stability
    try:
        Ks, innovation_cov = kalman_dot(innovation, V, C, R)
        assert np.allclose(innovation_cov, 0, atol=epsilon), \
            "Expected nearly zero innovation_cov with constructed singularity"
    except np.linalg.LinAlgError:
        pytest.fail("kalman_dot raised LinAlgError with nearly singular innovation_cov")


def test_kalman_dot_random_values():
    # Randomized test to ensure function works with arbitrary valid values
    n_keypoints = 5
    n_latents = 4

    innovation = np.random.randn(n_keypoints)
    V = np.random.randn(n_latents, n_latents)
    V = np.dot(V, V.T)  # Make V symmetric positive semi-definite
    C = np.random.randn(n_keypoints, n_latents)
    R = np.random.randn(n_keypoints, n_keypoints)
    R = np.dot(R, R.T)  # Make R symmetric positive semi-definite

    # Run kalman_dot
    Ks, innovation_cov = kalman_dot(innovation, V, C, R)

    # Check if innovation_cov is positive semi-definite (eigenvalues should be non-negative or close to zero)
    eigvals = np.linalg.eigvalsh(innovation_cov)
    assert np.all(eigvals >= -1e-8), "Expected innovation_cov to be positive semi-definite"
    assert Ks.shape == (n_latents,), f"Expected shape {(n_latents,)}, got {Ks.shape}"
    assert innovation_cov.shape == (n_keypoints, n_keypoints), \
        f"Expected shape {(n_keypoints, n_keypoints)}, got {innovation_cov.shape}"

    # Check that innovation_cov and Ks have finite values
    assert np.isfinite(innovation_cov).all(), "Expected finite values in innovation_cov"
    assert np.isfinite(Ks).all(), "Expected finite values in Ks"


def test_forward_pass_basic():
    # Set up basic test data
    T = 10
    n_keypoints = 5
    n_latents = 3

    y = np.random.randn(T, n_keypoints)
    m0 = np.random.randn(n_latents)
    S0 = np.eye(n_latents)
    C = np.random.randn(n_keypoints, n_latents)
    R = np.eye(n_keypoints)
    A = np.eye(n_latents)
    Q = np.eye(n_latents)
    ensemble_vars = np.abs(np.random.randn(T, n_keypoints))  # Variance should be non-negative

    # Run forward_pass
    mf, Vf, S, innovations, innovation_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)

    # Check output shapes
    assert mf.shape == (T, n_latents), f"Expected shape {(T, n_latents)}, got {mf.shape}"
    assert Vf.shape == (
    T, n_latents, n_latents), f"Expected shape {(T, n_latents, n_latents)}, got {Vf.shape}"
    assert S.shape == (
    T, n_latents, n_latents), f"Expected shape {(T, n_latents, n_latents)}, got {S.shape}"
    assert innovations.shape == (
    T, n_keypoints), f"Expected shape {(T, n_keypoints)}, got {innovations.shape}"
    assert innovation_cov.shape == (T, n_keypoints,
                                    n_keypoints), f"Expected shape {(T, n_keypoints, n_keypoints)}, got {innovation_cov.shape}"


def test_forward_pass_with_nan_values():
    # Test with some NaN values in y
    T = 10
    n_keypoints = 5
    n_latents = 3

    y = np.random.randn(T, n_keypoints)
    y[2, 1] = np.nan  # Insert NaN value
    m0 = np.random.randn(n_latents)
    S0 = np.eye(n_latents)
    C = np.random.randn(n_keypoints, n_latents)
    R = np.eye(n_keypoints)
    A = np.eye(n_latents)
    Q = np.eye(n_latents)
    ensemble_vars = np.abs(np.random.randn(T, n_keypoints))

    # Run forward_pass
    mf, Vf, S, innovations, innovation_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)

    # Check that non-NaN entries in y yield finite results in mf until the first NaN propagation
    found_nan_propagation = False
    for t in range(T):
        if np.isnan(y[t]).any():
            found_nan_propagation = True
            assert np.isnan(mf[t]).all(), f"Expected NaNs in mf at time {t}, found finite values"
        else:
            if found_nan_propagation:
                # Once NaNs are expected, allow them to propagate
                assert np.isnan(mf[t]).all(), f"Expected NaNs in mf at time {t} due to propagation, found finite values"
            else:
                # Check for finite values up until the first NaN propagation
                assert np.isfinite(mf[t]).all(), f"Expected finite values in mf at time {t}, found NaNs"

    # Ensure Vf and innovation_cov have finite values where possible
    assert np.isfinite(Vf).all(), "Non-finite values found in Vf"
    assert np.isfinite(innovation_cov).all(), "Non-finite values found in innovation_cov"


def test_forward_pass_single_sample():
    # Test with a single sample (edge case)
    T = 1
    n_keypoints = 5
    n_latents = 3

    y = np.random.randn(T, n_keypoints)
    m0 = np.random.randn(n_latents)
    S0 = np.eye(n_latents)
    C = np.random.randn(n_keypoints, n_latents)
    R = np.eye(n_keypoints)
    A = np.eye(n_latents)
    Q = np.eye(n_latents)
    ensemble_vars = np.abs(np.random.randn(T, n_keypoints))

    # Run forward_pass
    mf, Vf, S, innovations, innovation_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)

    # Check output shapes with a single sample
    assert mf.shape == (T, n_latents), f"Expected shape {(T, n_latents)}, got {mf.shape}"
    assert Vf.shape == (
    T, n_latents, n_latents), f"Expected shape {(T, n_latents, n_latents)}, got {Vf.shape}"
    assert S.shape == (
    T, n_latents, n_latents), f"Expected shape {(T, n_latents, n_latents)}, got {S.shape}"
    assert innovations.shape == (
    T, n_keypoints), f"Expected shape {(T, n_keypoints)}, got {innovations.shape}"
    assert innovation_cov.shape == (T, n_keypoints, n_keypoints), \
        f"Expected shape {(T, n_keypoints, n_keypoints)}, got {innovation_cov.shape}"


def test_forward_pass_zero_ensemble_vars():
    # Test with zero ensemble_vars to check stability
    T = 10
    n_keypoints = 5
    n_latents = 3

    y = np.random.randn(T, n_keypoints)
    m0 = np.random.randn(n_latents)
    S0 = np.eye(n_latents)
    C = np.random.randn(n_keypoints, n_latents)
    R = np.eye(n_keypoints)
    A = np.eye(n_latents)
    Q = np.eye(n_latents)
    ensemble_vars = np.zeros((T, n_keypoints))  # Ensemble vars set to zero

    # Run forward_pass
    mf, Vf, S, innovations, innovation_cov = forward_pass(y, m0, S0, C, R, A, Q, ensemble_vars)

    # Check if outputs are finite and correctly shaped
    assert np.isfinite(mf).all(), "Non-finite values found in mf with zero ensemble_vars"
    assert np.isfinite(Vf).all(), "Non-finite values found in Vf with zero ensemble_vars"
    assert np.isfinite(S).all(), "Non-finite values found in S with zero ensemble_vars"
    assert np.isfinite(
        innovations).all(), "Non-finite values found in innovations with zero ensemble_vars"
    assert np.isfinite(
        innovation_cov).all(), "Non-finite values found in innovation_cov with zero ensemble_vars"


def test_backward_pass_basic():
    # Set up basic test data
    T = 10
    n_keypoints = 5
    n_latents = 3

    y = np.random.randn(T, n_keypoints)
    mf = np.random.randn(T, n_latents)  # Should match n_latents
    Vf = np.random.randn(T, n_latents, n_latents)
    Vf = np.array([np.dot(v, v.T) for v in Vf])  # Make Vf positive semi-definite
    S = np.copy(Vf)  # Use S as the same structure as Vf
    A = np.eye(n_latents)

    # Run backward_pass
    ms, Vs, CV = backward_pass(y, mf, Vf, S, A)

    # Verify shapes of output arrays
    assert ms.shape == (T, n_latents), f"Expected shape {(T, n_latents)}, got {ms.shape}"
    assert Vs.shape == (T, n_latents, n_latents), f"Expected shape {(T, n_latents, n_latents)}, got {Vs.shape}"
    assert CV.shape == (T - 1, n_latents, n_latents), f"Expected shape {(T - 1, n_latents, n_latents)}, got {CV.shape}"

    # Check that ms, Vs, and CV contain finite values
    assert np.isfinite(ms).all(), "Non-finite values found in ms"
    assert np.isfinite(Vs).all(), "Non-finite values found in Vs"
    assert np.isfinite(CV).all(), "Non-finite values found in CV"


def test_backward_pass_with_nan_values():
    # Test with some NaN values in y
    T = 10
    n_keypoints = 5
    n_latents = 3

    y = np.random.randn(T, n_keypoints)
    y[2, 1] = np.nan  # Insert NaN value
    mf = np.random.randn(T, n_latents)  # Adjust shape to match n_latents
    Vf = np.random.randn(T, n_latents, n_latents)
    Vf = np.array([np.dot(v, v.T) for v in Vf])  # Make Vf positive semi-definite
    S = np.copy(Vf)
    A = np.eye(n_latents)

    # Run backward_pass
    ms, Vs, CV = backward_pass(y, mf, Vf, S, A)

    # Verify shapes of output arrays
    assert ms.shape == (T, n_latents), f"Expected shape {(T, n_latents)}, got {ms.shape}"
    assert Vs.shape == (T, n_latents, n_latents), f"Expected shape {(T, n_latents, n_latents)}, got {Vs.shape}"
    assert CV.shape == (T - 1, n_latents, n_latents), f"Expected shape {(T - 1, n_latents, n_latents)}, got {CV.shape}"

    # Check that ms, Vs, and CV contain finite values
    assert np.isfinite(ms).all(), "Non-finite values found in ms"
    assert np.isfinite(Vs).all(), "Non-finite values found in Vs"
    assert np.isfinite(CV).all(), "Non-finite values found in CV"


def test_backward_pass_single_timestep():
    # Test with only one timestep (edge case)
    T = 1
    n_keypoints = 5
    n_latents = 3

    y = np.random.randn(T, n_keypoints)
    mf = np.random.randn(T, n_latents)  # Adjust shape to match n_latents
    Vf = np.eye(n_latents)[None, :, :]  # Shape (1, n_latents, n_latents)
    S = np.copy(Vf)
    A = np.eye(n_latents)

    # Run backward_pass
    ms, Vs, CV = backward_pass(y, mf, Vf, S, A)

    # Verify shapes of output arrays
    assert ms.shape == (T, n_latents), f"Expected shape {(T, n_latents)}, got {ms.shape}"
    assert Vs.shape == (T, n_latents, n_latents), f"Expected shape {(T, n_latents, n_latents)}, got {Vs.shape}"
    assert CV.shape == (T - 1, n_latents, n_latents), f"Expected shape {(T - 1, n_latents, n_latents)}, got {CV.shape}"

    # Check that ms and Vs contain finite values
    assert np.isfinite(ms).all(), "Non-finite values found in ms"
    assert np.isfinite(Vs).all(), "Non-finite values found in Vs"


def test_backward_pass_singular_S_matrix():
    # Test with singular S matrix
    T = 10
    n_keypoints = 5
    n_latents = 3

    y = np.random.randn(T, n_keypoints)
    mf = np.random.randn(T, n_latents)  # Adjust shape to match n_latents
    Vf = np.random.randn(T, n_latents, n_latents)
    Vf = np.array([np.dot(v, v.T) for v in Vf])  # Make Vf positive semi-definite
    S = np.zeros((T, n_latents, n_latents))  # Singular S matrix (all zeros)
    A = np.eye(n_latents)

    # Run backward_pass and check stability
    try:
        ms, Vs, CV = backward_pass(y, mf, Vf, S, A)

        # Verify shapes of output arrays
        assert ms.shape == (T, n_latents), f"Expected shape {(T, n_latents)}, got {ms.shape}"
        assert Vs.shape == (
        T, n_latents, n_latents), f"Expected shape {(T, n_latents, n_latents)}, got {Vs.shape}"
        assert CV.shape == (T - 1, n_latents,
                            n_latents), f"Expected shape {(T - 1, n_latents, n_latents)}, got {CV.shape}"

        # Check for finite values in outputs, expecting NaNs or Infs due to singular S
        assert np.all(np.isfinite(ms)), "Non-finite values found in ms"
        assert np.all(np.isfinite(Vs)), "Non-finite values found in Vs"
        assert np.all(np.isfinite(CV)), "Non-finite values found in CV"

    except np.linalg.LinAlgError:
        pytest.fail("backward_pass failed due to singular S matrix")


def test_backward_pass_random_values():
    # Randomized test to ensure function works with arbitrary valid values
    T = 10
    n_keypoints = 6
    n_latents = 4

    y = np.random.randn(T, n_keypoints)
    mf = np.random.randn(T, n_latents)  # Adjust shape to match n_latents
    Vf = np.random.randn(T, n_latents, n_latents)
    Vf = np.array([np.dot(v, v.T) for v in Vf])  # Make Vf positive semi-definite
    S = np.copy(Vf)
    A = np.eye(n_latents)

    # Run backward_pass
    ms, Vs, CV = backward_pass(y, mf, Vf, S, A)

    # Verify shapes of output arrays
    assert ms.shape == (T, n_latents), f"Expected shape {(T, n_latents)}, got {ms.shape}"
    assert Vs.shape == (T, n_latents, n_latents), f"Expected shape {(T, n_latents, n_latents)}, got {Vs.shape}"
    assert CV.shape == (T - 1, n_latents, n_latents), f"Expected shape {(T - 1, n_latents, n_latents)}, got {CV.shape}"

    # Check that ms, Vs, and CV contain finite values
    assert np.isfinite(ms).all(), "Non-finite values found in ms"
    assert np.isfinite(Vs).all(), "Non-finite values found in Vs"
    assert np.isfinite(CV).all(), "Non-finite values found in CV"


def test_compute_nll_basic():
    # Set up basic test data
    T = 10
    n_coords = 3

    innovations = np.random.randn(T, n_coords)
    innovation_covs = np.array([np.eye(n_coords) for _ in range(T)])  # Identity matrices

    # Run compute_nll
    nll, nll_values = compute_nll(innovations, innovation_covs)

    # Check output types
    assert isinstance(nll, float), f"Expected nll to be float, got {type(nll)}"
    assert isinstance(nll_values, list), f"Expected nll_values to be list, got {type(nll_values)}"

    # Check nll_values length
    assert len(nll_values) == T, f"Expected length {T}, got {len(nll_values)}"

    # Check that all values in nll_values are positive
    assert all(v >= 0 for v in nll_values), "Expected all nll_values to be non-negative"


def test_compute_nll_with_nan_innovations():
    # Test with some NaN values in innovations
    T = 10
    n_coords = 3

    innovations = np.random.randn(T, n_coords)
    innovations[2, 1] = np.nan  # Insert NaN value
    innovation_covs = np.array([np.eye(n_coords) for _ in range(T)])

    # Run compute_nll
    nll, nll_values = compute_nll(innovations, innovation_covs)

    # Check nll_values length
    assert len(nll_values) == T - 1, f"Expected length {T - 1}, got {len(nll_values)}"
    # Check that nll is finite
    assert np.isfinite(nll), "Expected finite nll despite NaN in innovations"


def test_compute_nll_zero_innovation_covs():
    # Test with zero matrices for innovation_covs
    T = 5
    n_coords = 2

    innovations = np.random.randn(T, n_coords)
    innovation_covs = np.zeros((T, n_coords, n_coords))  # Zero matrices for innovation_covs

    # Run compute_nll
    nll, nll_values = compute_nll(innovations, innovation_covs, epsilon=1e-6)

    # Check nll is finite and values are positive due to epsilon regularization
    assert np.isfinite(nll), "Expected finite nll with zero innovation_covs"
    assert all(v >= 0 for v in nll_values), "Expected all nll_values to be non-negative"


def test_compute_nll_small_epsilon():
    # Test with a small epsilon to ensure stability with near-singular innovation_covs
    T = 10
    n_coords = 3

    innovations = np.random.randn(T, n_coords)
    # Make innovation_covs near-singular by making all elements small
    innovation_covs = np.full((T, n_coords, n_coords), 1e-8)

    # Run compute_nll with a very small epsilon
    nll, nll_values = compute_nll(innovations, innovation_covs, epsilon=1e-10)

    # Check that nll is finite
    assert np.isfinite(nll), "Expected finite nll with small epsilon"
    assert len(nll_values) == T, f"Expected length {T}, got {len(nll_values)}"
    # Ensure all values in nll_values are positive
    assert all(v >= 0 for v in nll_values), "Expected all nll_values to be non-negative"


def test_compute_nll_random_values():
    # Randomized test with arbitrary valid values
    T = 8
    n_coords = 4

    innovations = np.random.randn(T, n_coords)
    innovation_covs = np.random.randn(T, n_coords, n_coords)
    # Make innovation_covs positive semi-definite
    innovation_covs = np.array([np.dot(c, c.T) for c in innovation_covs])

    # Run compute_nll
    nll, nll_values = compute_nll(innovations, innovation_covs)

    # Check nll and nll_values length
    assert isinstance(nll, float), f"Expected nll to be float, got {type(nll)}"
    assert len(nll_values) == T, f"Expected length {T}, got {len(nll_values)}"
    # Ensure finite values
    assert np.isfinite(nll), "Expected finite nll"
    assert all(
        np.isfinite(nll_val) for nll_val in nll_values), "Expected all nll_values to be finite"


def test_jax_ensemble_basic():
    # Basic test data
    n_models = 4
    n_timepoints = 5
    n_keypoints = 3
    markers_3d_array = np.random.rand(n_models, n_timepoints, n_keypoints * 3)

    # Run jax_ensemble in median mode
    ensemble_preds, ensemble_vars, keypoints_avg_dict = jax_ensemble(markers_3d_array, mode='median')

    # Check output shapes
    assert ensemble_preds.shape == (n_timepoints, n_keypoints, 2), \
        f"Expected shape {(n_timepoints, n_keypoints, 2)}, got {ensemble_preds.shape}"
    assert ensemble_vars.shape == (n_timepoints, n_keypoints, 2), \
        f"Expected shape {(n_timepoints, n_keypoints, 2)}, got {ensemble_vars.shape}"
    assert len(keypoints_avg_dict) == n_keypoints * 2, \
        f"Expected {n_keypoints * 2} entries in keypoints_avg_dict, got {len(keypoints_avg_dict)}"

def test_jax_ensemble_median_mode():
    # Test median mode
    n_models = 4
    n_timepoints = 5
    n_keypoints = 3
    markers_3d_array = np.random.rand(n_models, n_timepoints, n_keypoints * 3)

    # Run jax_ensemble
    ensemble_preds, ensemble_vars, _ = jax_ensemble(markers_3d_array, mode='median')

    # Check that ensemble_preds and ensemble_vars are finite
    assert jnp.isfinite(ensemble_preds).all(), "Expected finite values in ensemble_preds"
    assert jnp.isfinite(ensemble_vars).all(), "Expected finite values in ensemble_vars"

def test_jax_ensemble_mean_mode():
    # Test mean mode
    n_models = 4
    n_timepoints = 5
    n_keypoints = 3
    markers_3d_array = np.random.rand(n_models, n_timepoints, n_keypoints * 3)

    # Run jax_ensemble in mean mode
    ensemble_preds, ensemble_vars, _ = jax_ensemble(markers_3d_array, mode='mean')

    # Check that ensemble_preds and ensemble_vars are finite
    assert jnp.isfinite(ensemble_preds).all(), "Expected finite values in ensemble_preds"
    assert jnp.isfinite(ensemble_vars).all(), "Expected finite values in ensemble_vars"

def test_jax_ensemble_confidence_weighted_mean_mode():
    # Test confidence-weighted mean mode
    n_models = 4
    n_timepoints = 5
    n_keypoints = 3
    markers_3d_array = np.random.rand(n_models, n_timepoints, n_keypoints * 3)

    # Run jax_ensemble in confidence_weighted_mean mode
    ensemble_preds, ensemble_vars, _ = jax_ensemble(markers_3d_array, mode='confidence_weighted_mean')

    # Check that ensemble_preds and ensemble_vars are finite
    assert jnp.isfinite(ensemble_preds).all(), "Expected finite values in ensemble_preds"
    assert jnp.isfinite(ensemble_vars).all(), "Expected finite values in ensemble_vars"

def test_jax_ensemble_unsupported_mode():
    # Test that unsupported mode raises ValueError
    n_models = 4
    n_timepoints = 5
    n_keypoints = 3
    markers_3d_array = np.random.rand(n_models, n_timepoints, n_keypoints * 3)

    with pytest.raises(ValueError, match="averaging not supported"):
        jax_ensemble(markers_3d_array, mode='unsupported')


if __name__ == "__main__":
    pytest.main([__file__])