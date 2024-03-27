# Scripts

This document is a general overview of the inputs, workflow, and outputs for the example scripts in `scripts/`.
It covers the overlapping code across the four examples and notes where they differ. See [Command-Line Arguments](command-line_arguments.md) for usage.

## Input

The input is a directory of csv files containing data in either DLC or LP output form.
LP data is converted to DLC via `convert_lp_dlc()` in [utils.py](../eks/utils.py). Input data must 
have three headers, scorer, bodyparts, and coords. The name of the scorer will be replaced in the
output as specified by `tracker_name`, a parameter in the main EKS smoother function called by the script.
The body part names must be identical to the bodypart command-line arguments.
The coords must take the form x, y, and likelihood. The following is an example taken from
[IBL-paw](../data/ibl-paw/3f859b5c-e73a-4044-b49e-34bb81e96715.left.rng=0.csv), showing the necessary column
headers:

| scorer               | heatmap_mhcrnn_tracker | heatmap_mhcrnn_tracker | heatmap_mhcrnn_tracker | heatmap_mhcrnn_tracker | heatmap_mhcrnn_tracker | heatmap_mhcrnn_tracker |
|----------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| bodyparts            | paw_l                  | paw_l                  | paw_l                  | paw_r                  | paw_r                  | paw_r                  |
| coords               | x                      | y                      | likelihood             | x                      | y                      | likelihood             |


## Workflow

1. **Argument Parsing**:
   - Parse [command-line arguments](command-line_arguments.md) to specify input directories and parameters.

2. **EKS Execution**:
   - Check if the provided CSV directory exists.
   - Load CSV files containing marker predictions and convert them into the correct format.
   - Apply Ensemble Kalman Smoothing (EKS) to each keypoint, iterating through each camera view in the case of multiple cameras
   - Each example script calls one of the specialized [EKS smoothers](eks_smoothers.md) specific to that use-case.
   - Save the EKS results to a CSV file.

3. **Plotting Results**:
   - Select an example keypoint from the provided list.
   - Plot individual model predictions and EKS-smoothed predictions for `x`, `y`, `likelihood`, and `zscore`.
   - Save the plot as a PDF file.

## Output

The script generates two main outputs:
- A CSV file containing the EKS-smoothed results.
- A PDF file containing visualizations of the EKS results for an example keypoint.

## TODO
Standardize where the tracker_name is stored (currently in pupil_smoother as a parameter,
hard-coded in multiview_pca_smoother, and hard-coded in utils.py function make_dlc_pandas_index())
Improve visualizations (video?)