# Command-Line Arguments

This document provides an overview of command-line arguments featured in the provided example scripts (in `scripts/`)
for running the EKS.

The following is the general syntax for running an example script in bash:

```bash
python scripts/<your_script> --csv-dir <csv-directory> --<optional-arg1> <params1> --<optional-arg2> <params2> ...
```
The script is run as a python executable. Arguments are parsed using the `argparse`
[Python module](https://docs.python.org/3/library/argparse.html), as seen in each of
the example scripts.

Arguments fall into two categories:
## File I/O Arguments
These arguments dictate the file directories for reading and writing data, and are present in all example scripts.
- `--csv-dir <csv_directory>` String specifying read-in directory containing CSV files used as input data.
- `--save-dir <save_directory>` String specifying write-to directory (csv-dir by default).

## Script-specific Arguments
These arguments are specific to the example script (e.g. `eks/singleview_smoother.py`) being used, which is
smoother-dependent. The following are the arguments found in the provided examples in `scripts/`.
### [Single-Camera](../scripts/singlecam_example.py)
- `--bodypart-list <bodypart_1> <bodypart_2> ...` List of body parts to be tracked by the EKS. Names must have corresponding csv column names in the input data.
- `--s <smooth_param>` Float specifying the extent of smoothing to be done. Smoothing increases as param **decreases** (range 0.01-20, 10 by default)
### [Multi-Camera](../scripts/multicam_example.py)
- `--bodypart-list <bodypart_1> <bodypart_2> ...` List of body parts to be tracked by the EKS. Names must have corresponding csv column names in the input data.
- `--camera-names <camera_1> <camera_2> ...` List of camera views.
- `--s <smooth_param>` Float specifying the extent of smoothing to be done. Smoothing increases as param **decreases** (range 0.01-20, 0.01 by default)
- `--quantile_keep_pca <min_variance%>` Float specifying the percentage of points kept for multi-view PCA. Selectivity increases as param increases (range 0-100, 25 by default)
### [IBL Pupil](../scripts/pupil_example.py)
- `--diameter-s <smooth_param>` Float specifying the extent of smoothing to be done for diameter. Smoothing increases as param **increases** (range 0-1 exclusive, 0.9999 by default)
- `--com-s <smooth_param>` Float specifying the extent of smoothing to be done for center of mass. Smoothing increases as param **increases** (range 0-1 exclusive, 0.999 by default)
### [IBL Paw (multiple asynchronous view)](../scripts/multiview_paw_example.py)
- `--s <smooth_param>` Float specifying the extent of smoothing to be done. Smoothing increases as param **decreases** (range 0.01-20, 0.01 by default)
- `--quantile_keep_pca <min_variance%>` Float specifying the percentage of points kept for multi-view PCA. Selectivity increases as param increases (range 0-100, 25 by default)

The following table summarizes the script-specific arguments featured in each of the example scripts.

| Argument\Script       | [Single-Camera](../scripts/singlecam_example.py) | [Multi-Camera](../scripts/multicam_example.py) | [IBL Pupil](../scripts/pupil_example.py) | [IBL Paw](../scripts/multiview_paw_example.py) |
|-----------------------|---------------|--------------|-----------|---------------------|
| `--bodypart-list`     | ✓             | ✓            |           |                     |
| `--camera-names`      |               | ✓            |           |                     |
| `--s`                 | ✓             | ✓            |           | ✓                   |
| `--quantile_keep_pca` |               | ✓            |           | ✓                   |
| `--diameter-s`        |               |              | ✓         |                     |
| `--com-s`             |               |              | ✓         |                     |

## TODO & Takeaways
- Script-specific args are somewhat overlapping, suggesting refactoring will be useful.
- Probably good to keep the smoothing param consistent as far as positive/negative correlation to smoothing amount (for IBL Pupil)