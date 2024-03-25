# Command-Line Arguments

This document provides an overview of command-line arguments featured in the provided example scripts (in `scripts/`)
for running the EKS.

The following is the general syntax for running an example script in bash:

```bash
python scripts/<your_script> --csv-dir <csv-directory> --<optional-arg1> <params1> --<optional-arg2> <params2> ...
```
The script is run as a python executable, with various arguments that fall into two categories:
## File I/O Arguments
These arguments dictate the file directories for reading and writing data, and are present in all example scripts.
- `--csv-dir <csv_directory>` String specifying read-in directory containing CSV files used as input data.
- `--save-dir <save_directory>` String specifying write-to directory (default is csv-dir).

## Script-specific Arguments
These arguments are specific to the example script (e.g. `eks/singleview_smoother.py`) being used, which is
smoother-dependent. The following are the arguments found in the provided examples in `scripts/`.
### [Single-Camera](../scripts/singlecam_example.py)
- `--bodypart-list <bodypart_1> <bodypart_2> ...` List of body parts to be tracked by the EKS.
- `--s <smooth_param>` Float specifying the amount of smoothing to be done
### [Multi-Camera](../scripts/multicam_example.py)
### [IBL Pupil](../scripts/pupil_example.py)
### [IBL Paw (multiple asynchronous view)](../scripts/multiview_paw_example.py)