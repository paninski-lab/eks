# Workflow Overview

This documentation provides an overview of the high-level workflow involved in processing 
single-camera datasets using `singlecam_example.py`. `singlecam_example.py` is currently the most
up-to-date script, incorporating efficient optimization routines for finding a suitable smoothing
parameter given data, and is the most useful script to walk through in order to provide a high-level
understanding of the EKS workflow.

Here, we will progress through the high-level workflow of `singlecam_example.py`. Further details on
key functions `ensemble_kalman_smoother_singlecam` and `singlecam_optimize_smooth` are provided as
well.

---

## singlecam_example.py

### Overview

The `singlecam_example.py` script demonstrates how to process and smooth single-camera datasets. 
It includes steps to handle input/output operations, format data, and apply the Ensemble Kalman 
Smoother (EKS).

### Workflow Steps

1. **Collect User-Provided Arguments**:
    - Define the `smoother_type` as 'singlecam'.
    - Parse command-line arguments using `handle_parse_args(smoother_type)`.
    - Extract and set various input parameters such as `input_dir`, `data_type`, `save_dir`, 
    - `save_filename`, `bodypart_list`, `s`, `s_frames`, and `blocks`.

2. **Load and Format Input Data**:
    - Use `format_data` to read and format input files, and prepare an empty DataFrame for output.
    - If `bodypart_list` is not provided, use keypoint names from the input data.
    - Print the keypoints being processed.

3. **Convert Input Data to 3D Array**:
    - Convert the list of DataFrames to a 3D NumPy array using `np.stack`.
    - Map keypoint names to their respective indices in the DataFrames.
    - Crop the 3D array to include only the columns corresponding to the specified body parts 
    - (`_x`, `_y`, `_likelihood`).

4. **Apply Ensemble Kalman Smoother**:
    - Call `ensemble_kalman_smoother_singlecam` from `singlecam_smoother.py`
   with the prepared 3D array and other arguments to 
   obtain smoothed results (`df_dicts`, `s_finals`).

5. **Save Smoothed Results**:
    - For each body part, convert the resulting DataFrames to CSV files.
    - Save the output DataFrame as a CSV file in the specified directory.

6. **Plot Results**:
    - Use `plot_results` to visualize the smoothed data.
    - Plot the results for a specific keypoint (`keypoint_i`).

---


### Key Function: `ensemble_kalman_smoother_singlecam`

(from `eks/singlecam_smoother.py`)

This function performs Ensemble Kalman Smoothing on 3D marker data from a single camera. It takes as input a 3D array of marker data, a list of body parts, smoothing parameters, and frames, and returns dataframes with smoothed predictions, final smoothing parameters, and Negative Log-Likelihood (NLL) values.

#### Parameters:

- **`markers_3d_array` (np.ndarray)**: A 3D array of marker data with dimensions corresponding to time frames, body parts, and coordinates (x, y, z).
- **`bodypart_list` (list)**: A list of body parts for which the data is being processed.
- **`smooth_param` (float)**: A parameter controlling the smoothing process.
- **`s_frames` (list)**: A list of frames used in the smoothing process.
- **`blocks` (list)**: Optional. A list of blocks for segmenting the data (default is an empty list).
- **`ensembling_mode` (str)**: The mode used for ensembling the data (default is 'median').
- **`zscore_threshold` (float)**: The Z-score threshold for outlier detection (default is 2).

#### Returns:

- **`tuple`**: A tuple containing:
  - Dataframes with smoothed predictions.
  - Final smoothing parameters (per keypoint).
  - NLL values (used for finding the ideal smoothing parameter value)

### Detailed Steps:

1. **Initialization**:
   - Extract the total number of frames (`T`) and the number of keypoints (`n_keypoints`) from the `markers_3d_array`.
   - Define the number of coordinates (`n_coords`) as 2 (x and y).

2. **Ensemble Statistics**:
   - Compute ensemble statistics by calling `jax_ensemble` with the marker data and ensembling mode.
   - Extract ensemble predictions (`ensemble_preds`), variances (`ensemble_vars`), and average keypoints (`keypoints_avg_dict`).

3. **Adjust Observations**:
   - Calculate mean and adjusted observations by calling `adjust_observations` with the average keypoints, number of keypoints, and ensemble predictions.
   - Obtain `mean_obs_dict`, `adjusted_obs_dict`, and `scaled_ensemble_preds`.

4. **Initialize Kalman Filter**:
   - Initialize Kalman filter values by calling `initialize_kalman_filter` with the scaled ensemble predictions, adjusted observations, and number of keypoints.
   - Obtain initial means (`m0s`), covariances (`S0s`), state transition matrices (`As`), covariance matrices (`cov_mats`), observation matrices (`Cs`), observation covariances (`Rs`), and observations (`ys`).

5. **Smoothing**:
   - Perform the main smoothing function by calling `singlecam_optimize_smooth` with the initialized values, ensemble variances, frames, smoothing parameter, and blocks.
   - Obtain final smoothing parameters (`s_finals`), means (`ms`), and covariances (`Vs`).

6. **Process Each Keypoint**:
   - Initialize arrays for smoothed means (`y_m_smooths`), variances (`y_v_smooths`), and predicted arrays (`eks_preds_array`).
   - Loop through each keypoint to compute smoothed predictions and variances, adjust predictions based on mean observations, and compute Z-scores using `eks_zscore`.

7. **Final Cleanup**:
   - Create a pandas DataFrame for each keypoint with smoothed predictions, variances, and Z-scores.
   - Append each DataFrame to a list (`dfs`) and a dictionary (`df_dicts`).

8. **Return Results**:
   - Return a tuple containing the dictionary of DataFrames and the final smoothing parameters.

---


### Key Function: `singlecam_optimize_smooth`

This function optimizes the smoothing parameter and uses the result to run the Kalman filter-smoother. It takes in various parameters related to covariance matrices, observations, and initial states, and returns the final smoothing parameters, smoothed means, and smoothed covariances.

#### Parameters:

- **`cov_mats` (np.ndarray)**: Covariance matrices.
- **`ys` (np.ndarray)**: Observations with shape (keypoints, frames, coordinates), where coordinate is usually 2.
- **`m0s` (np.ndarray)**: Initial mean state.
- **`S0s` (np.ndarray)**: Initial state covariance.
- **`Cs` (np.ndarray)**: Measurement function.
- **`As` (np.ndarray)**: State-transition matrix.
- **`Rs` (np.ndarray)**: Measurement noise covariance.
- **`ensemble_vars` (np.ndarray)**: Ensemble variances.
- **`s_frames` (list)**: List of frames.
- **`smooth_param` (float)**: Smoothing parameter.
- **`blocks` (list)**: List of blocks for segmenting the data.
- **`maxiter` (int)**: Maximum number of iterations for optimization (default is 1000).

#### Returns:

- **`tuple`**: A tuple containing:
  - Final smoothing parameters.
  - Smoothed means.
  - Smoothed covariances.
  - Negative log-likelihoods.
  - Negative log-likelihood values.

### Detailed Steps:

1. **Initialization**:
   - Extract the number of keypoints (`n_keypoints`) from the `ys` array.
   - Initialize an empty list for final smoothing parameters (`s_finals`).
   - If no blocks are provided, create a block for each keypoint.

2. **Device Check**:
   - Check if a GPU is available for parallel processing. If available, use the GPU for parallel smoothing parameter optimization. Otherwise, use the CPU for sequential optimization.

3. **Define Loss Functions**:
   - Define `nll_loss_parallel_scan` for GPU usage and `nll_loss_sequential_scan` for CPU usage. Both functions ensure positivity by taking the exponential of the smoothing parameter and call the appropriate smoothing function (`singlecam_smooth_min_parallel` or `singlecam_smooth_min`).

4. **Smooth Parameter Optimization**:
   - If a `smooth_param` is provided, use it directly. Otherwise, initialize guesses for each keypoint using `compute_initial_guesses` and crop the frames using `crop_frames`.
   - Optimize the negative log-likelihood for each block of keypoints:
     - Initialize the smoothing parameter (`s_init`) with a positive guess.
     - Set up the optimizer using `optax.adam`.
     - Select the relevant subsets of the input arrays for the current block.
     - Define a `step` function to perform optimization steps.
     - Iterate the optimization process until convergence or the maximum number of iterations is reached.

5. **Final Smooth**:
   - After optimization, perform a final forward-backward pass with the optimized smoothing parameters by calling `final_forwards_backwards_pass`.

6. **Return Results**:
   - Return a tuple containing the final smoothing parameters, smoothed means, and smoothed covariances.

