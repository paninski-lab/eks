# EKS Smoothers (WIP)

This document covers the three EKS smoothers for Single-view, Multi-view, and Pupil use-cases,
working through their main functions while noting their similarities and differences.

## Overview

Smoothers take in the DataFrame-formatted marker data from [scripts](scripts.md), and utilizes
functions in [ensemble-kalman.py](../eks/ensemble_kalman.py) to run the EKS on the input data, using the
smoothing parameter in the state-covariance matrix to improve the accuracy of the smoothed predictions.
It returns a new DataFrame containing the smoothed markers while retaining the input format, using
`make_dlc_pandas_index()` from [utils.py](../eks/utils.py).

## Function Details

All smoothers have a main function beginning with `ensemble_kalman_...` (the Multi-view smoother
has two of these, one for multi-camera and one for IBL-paw). For instance,
[singleview_smoother.py](../eks/singleview_smoother.py) has the function:
```python
def ensemble_kalman_smoother_single_view(
        markers_list, keypoint_ensemble, smooth_param, ensembling_mode='median', zscore_threshold=2, verbose=False):
```

These functions apply the Ensemble Kalman Smoother (EKS) to smooth marker data from multiple
ensemble members for each view.

#### Parameters
- `markers_list` : List of List of DataFrame 
  - Contains the formatted DataFrames containing predictions from each ensemble member. The 
multi-view smoother has an additional parameter for each camera view.
- `keypoint_ensemble` : str
  - The name of the keypoint to be ensembled and smoothed. Parameter `keypoint_names`
taken as a list of keypoints instead in the multi-view and pupil smoother.
- `smooth_param` : float
  - See [command-line_arguments.md](command-line_arguments.md) for information on the smoothing parameter.
The pupil smoother takes in parameter `state_transition_matrix` instead, which is built from the two smoothing
parameters `diameter-s` and `com-s` in the [pupil example script](../scripts/pupil_example.py).
- `ensembling_mode` : str, optional
  - The function used for ensembling. Options are 'mean', 'median', or 'confidence_weighted_mean',
median by default. The parameter does not exist for [pupil_smoother](../eks/pupil_smoother) and instead defaults to median.
- `zscore_threshold` : float, optional
  - Minimum standard deviation threshold to reduce the effect of low ensemble standard deviation on a z-score metric. Default is 2.
- `verbose` : bool, optional
  - If True, progress will be printed for the user.

#### Returns
- 'keypoint_df': DataFrame containing smoothed markers for one keypoint. Same format as input DataFrames.