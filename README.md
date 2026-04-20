# EKS

![GitHub](https://img.shields.io/github/license/paninski-lab/eks)
![PyPI](https://img.shields.io/pypi/v/ensemble-kalman-smoother)
![PyPI Downloads](https://static.pepy.tech/badge/ensemble-kalman-smoother/week)

This repo contains code to run an Ensemble Kalman Smoother (EKS) for improving pose estimation outputs.

The EKS uses a Kalman smoother to ensemble and smooth pose estimation outputs as a post-processing
step after multiple model predictions have been generated, resulting in a more robust output:

![](assets/crim13_singlecam.gif)

For more details see [Biderman, Whiteway et al. 2024, Nature Methods](https://rdcu.be/dLP3z).

---

## Installation

We offer two methods for installing the `eks` package:
* Method 1, `github+conda`: this is the preferred installation method and will give you access to example data
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
git clone https://github.com/paninski-lab/eks
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

For more information on individual modules and their usage, see [Requirements](docs/requirements.md).

### Method 2: pip

You can also install the `eks` package using the Python Package Index (PyPI):
```
python3 -m pip install ensemble-kalman-smoother
```
Note that you will not have access to the example data with the pip install option.

## Usage

After installation, the `eks` command is available in your environment. Run `eks --help` to see
available subcommands, or `eks <subcommand> --help` for full argument details for any subcommand.

### Single-camera datasets

The `singlecam` subcommand runs EKS for standard single-camera setups. Any of the provided
datasets are compatible; below we use `data/ibl-pupil` as an example.

```console
eks singlecam --input-dir ./data/ibl-pupil --make-plot
```

### Multi-camera datasets

The `multicam` subcommand supports two modes depending on whether camera calibration is available.
Pose predictions should be stored in a separate CSV file per camera.

#### Without calibration (linear EKS)

Example data in `data/mirror-mouse-separate` contains two-view mouse video with cameras named
`top` and `bot`:

```console
eks multicam --input-dir ./data/mirror-mouse-separate --bodypart-list paw1LH paw2LF paw3RF paw4RH --camera-names top bot --make-plot
```

#### With calibration (nonlinear EKS)

Calibration data must be stored in `.toml` files using the
[Anipose](https://anipose.readthedocs.io/) format.
Example data in `data/fly` contains multi-view fly video with cameras `Cam-A`, `Cam-B`, and
`Cam-C`, along with a `calibration.toml` file:

```console
eks multicam --input-dir ./data/fly --bodypart-list L1A L1B --camera-names Cam-A Cam-B Cam-C --calibration ./data/fly/calibration.toml --make-plot
```

### Mirrored multi-camera datasets

The `mirrored-multicam` subcommand handles setups where pose predictions for all cameras are
stored in a single CSV file. For example, a body part `nose_tip` with cameras `top`, `bottom`,
and `side` should have columns named `nose_tip_top`, `nose_tip_bottom`, and `nose_tip_side`.
Example data in `data/mirror-mouse` contains a two-view mouse video with cameras `top` and `bot`:

```console
eks mirrored-multicam --input-dir ./data/mirror-mouse --bodypart-list paw1LH paw2LF paw3RF paw4RH --camera-names top bot --make-plot
```

### IBL pupil dataset

The `ibl-pupil` subcommand expects an `--input-dir` containing Lightning Pose or DLC model
predictions:

```console
eks ibl-pupil --input-dir ./data/ibl-pupil --make-plot
```

### IBL paw dataset (multiple asynchronous views)

The `ibl-paw` subcommand expects an `--input-dir` containing Lightning Pose or DLC model
predictions for the left and right camera views, as well as timestamp files to align the two
cameras:

```console
eks ibl-paw --input-dir ./data/ibl-paw --make-plot
```

### Authors

* [Cole Hurwitz](https://github.com/colehurwitz)
* [Keemin Lee](https://github.com/keeminlee)
* [Amol Pasarkar](https://github.com/apasarkar)
* [Matt Whiteway](https://github.com/themattinthehatt)
* [Spirit of claude]
