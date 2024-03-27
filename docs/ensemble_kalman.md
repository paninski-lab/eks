# Ensemble Kalman Smoother

## ensemble
```python
def ensemble(markers_list, keys, mode='median'):
```

Computes ensemble median (or mean) and variance of a list of DLC marker dataframes.

### Args:
- `markers_list`: list
  - List of DLC marker dataframes.
- `keys`: list
  - List of keys in each marker dataframe.
- `mode`: string, optional (default: 'median')
  - Averaging mode which includes 'median', 'mean', or 'confidence_weighted_mean'.

### Returns:
- `ensemble_preds`: np.ndarray
  - Shape: (samples, n_keypoints)
- `ensemble_vars`: np.ndarray
  - Shape: (samples, n_keypoints)
- `ensemble_stacks`: np.ndarray
  - Shape: (n_models, samples, n_keypoints)
- `keypoints_avg_dict`: dict
  - Keys: marker keypoints
  - Values: Shape (samples)
- `keypoints_var_dict`: dict
  - Keys: marker keypoints
  - Values: Shape (samples)
- `keypoints_stack_dict`: dict(dict)
  - Keys: model_ids
  - Values: Shape (samples)

## filtering_pass
```python
def filtering_pass(y, m0, S0, C, R, A, Q, ensemble_vars):
```

Implements the Kalman filter.

### Args:
- `y`: np.ndarray
  - Shape: (samples, n_keypoints)
- `m0`: np.ndarray
  - Shape: (n_latents)
- `S0`: np.ndarray
  - Shape: (n_latents, n_latents)
- `C`: np.ndarray
  - Shape: (n_keypoints, n_latents)
- `R`: np.ndarray
  - Shape: (n_keypoints, n_keypoints)
- `A`: np.ndarray
  - Shape: (n_latents, n_latents)
- `Q`: np.ndarray
  - Shape: (n_latents, n_latents)
- `ensemble_vars`: np.ndarray
  - Shape: (samples, n_keypoints)

### Returns:
- `mf`: np.ndarray
  - Shape: (samples, n_keypoints)
- `Vf`: np.ndarray
  - Shape: (samples, n_latents, n_latents)
- `S`: np.ndarray
  - Shape: (samples, n_latents, n_latents)

## kalman_dot
```python
def kalman_dot(array, V, C, R):
```

Helper function for matrix multiplication used in the Kalman filter.

### Args:
- `array`: np.ndarray
- `V`: np.ndarray
- `C`: np.ndarray
- `R`: np.ndarray

### Returns:
- `K_array`: np.ndarray

## smooth_backward
```python
def smooth_backward(y, mf, Vf, S, A, Q, C):
```

Implements Kalman smoothing backwards.

### Args:
- `y`: np.ndarray
  - Shape: (samples, n_keypoints)
- `mf`: np.ndarray
  - Shape: (samples, n_keypoints)
- `Vf`: np.ndarray
  - Shape: (samples, n_latents, n_latents)
- `S`: np.ndarray
  - Shape: (samples, n_latents, n_latents)
- `A`: np.ndarray
  - Shape: (n_latents, n_latents)
- `Q`: np.ndarray
  - Shape: (n_latents, n_latents)
- `C`: np.ndarray
  - Shape: (n_keypoints, n_latents)

### Returns:
- `ms`: np.ndarray
  - Shape: (samples, n_keypoints)
- `Vs`: np.ndarray
  - Shape: (samples, n_latents, n_latents)
- `CV`: np.ndarray
  - Shape: (samples, n_latents, n_latents)

## eks_zscore
```python
def eks_zscore(eks_predictions, ensemble_means, ensemble_vars, min_ensemble_std=2):
```

Computes the z-score between EKS prediction and the ensemble for a single keypoint.

### Args:
- `eks_predictions`: list
  - EKS prediction for each coordinate (x and y) for a single keypoint - Shape: (samples, 2)
- `ensemble_means`: list
  - Ensemble mean for each coordinate (x and y) for a single keypoint - Shape: (samples, 2)
- `ensemble_vars`: string
  - Ensemble variance for each coordinate (x and y) for a single keypoint - Shape: (samples, 2)
- `min_ensemble_std`: float, optional (default: 2)
  - Minimum standard deviation threshold to reduce the effect of low ensemble standard deviation.

### Returns:
- `z_score`: np.ndarray
  - Z-score for each time point - Shape: (samples, 1)