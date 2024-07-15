# EKS
This repo contains code to run an Ensemble Kalman Smoother (EKS) for improving pose estimation outputs.

The EKS uses a Kalman smoother to ensemble and smooth pose estimation outputs as a post-processing
step after multiple model predictions have been generated, resulting in a more robust output:

![](assets/crim13_singlecam.gif)

For more details see [Biderman, Whiteway et al. 2024, Nature Methods](https://rdcu.be/dLP3z).

---

## Installation

We offer two methods for installing the `eks` package:
* Method 1, `github+conda`: this is the preferred installation method and will give you access to example scripts and data
* Method 2, `pip`: this option is intended for non-interactive environments, such as remote servers.

For both installation methods we recommend using
[conda](https://docs.anaconda.com/free/anaconda/install/index.html)
to create a new environment in which this package and its dependencies will be installed:

```
conda create --name eks python=3.10
```

Activate the new environment:
```
conda activate eks
```

Make sure you are in the activated environment during the Lightning Pose installation.

### Method 1: github+conda

First you'll have to install the `git` package in order to access the code on github. 
Follow the directions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
for your specific OS.
Then, in the command line, navigate to where you'd like to install the `eks` package and move 
into that directory:
```
git clone https://github.com/colehurwitz/eks
cd eks
```

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main `eks` directory:

```
pip install -e .
```

If you wish to install the developer version of the package, run installation like this:
```
pip install -e ".[dev]"
```

For more information on individual modules and their usage, see [Requirements](docs/requirements.md)

### Method 2: pip

You can also install the `eks` package using the Python Package Index (PyPI):
```
python3 -m pip install ensemble-kalman-smoother
```
Note that you will not have access to the example data or example scripts with the pip install 
option.

### Note: Using GPU for fast parallel-scan
As of now, EKS singlecam features a jitted parallel scan implementation for quickly optimizing the
smoothing parameter (notably for larger datasets of 10,000+ frames). In order to utilize parallel scan,
you will need to have a cuda environment with jax enabled. Further instructions can be found in the [jax
docs](https://jax.readthedocs.io/en/latest/installation.html).
-------------------------------

## Example scripts

We provide several example datasets and fitting scripts to illustrate use of the package. See
[Command-Line Arguments](docs/command-line_arguments.md) for more information on arguments, 
including optional flags and defaults. We recommend starting with the first of four scripts outlined
below, `singlecam_example.py`, following along with the [Singlecam Overview](docs/singlecam_overview.md)
if a deeper understanding of EKS is desired.

### Single-camera datasets
The `singlecam_example.py` script demonstrates how to run the EKS code for standard single-camera
setups. 
Any of the provided datasets are compatible with this script; below we'll use `data/ibl-pupil` as
our example. 
To run the EKS on the example data, execute the following command from inside this repo:

```console 
python scripts/singlecam_example.py --input-dir ./data/ibl-pupil
```

The singlecam script is currently the most up-to-date script with the greatest number of feature
implementations, including fast smoothing parameter auto-tuning using GPU-driven parallelization.
[Here](docs/singlecam_overview.md) is a detailed overview of the workflow.
 
### Multi-camera datasets
The `multicam_example.py` script demonstrates how to run the EKS code for multi-camera
setups where the pose predictions for a given model are all stored in a single csv file. 
For example, if there is a body part names `nose_tip` and three cameras named 
`top`, `bottom`, and `side`, then the csv file should have columns named
`nose_tip_top`, `nose_tip_bottom`, and `nose_tip_side`.
We provide example data in the `data/mirror-mouse` directory inside this repo, 
for a two-view video of a mouse with cameras named `top` and `bot`. 
To run the EKS on the example data provided, execute the following command from inside this repo:

```console 
python scripts/multicam_example.py --input-dir ./data/mirror-mouse --bodypart-list paw1LH paw2LF paw3RF paw4RH --camera-names top bot
```

### IBL pupil dataset
The `pupil_example.py` script requires a `input-dir` which contains lightning-pose or DLC 
model predictions. 
To run this script on the example data provided, execute the following command from inside this repo:

```console 
python scripts/ibl_pupil_example.py --input-dir ./data/ibl-pupil
```

### IBL paw dataset (multiple asynchronous views)
The `multiview_paw_example.py` script requires a `input-dir` which contains lightning-pose 
or DLC model predictions for the left and right camera views, as well as timestamp files to align 
the two cameras. 
To run this script on the example data provided, execute the following command from inside this repo:

```console 
python scripts/ibl_paw_multiview_example.py --input-dir ./data/ibl-paw
```

### Authors

Cole Hurwitz

Keemin Lee

Amol Pasarkar

Matt Whiteway

