# EKS

[![Discord](https://img.shields.io/discord/1103381776895856720)](https://discord.gg/tDUPdRj4BM)
![GitHub](https://img.shields.io/github/license/paninski-lab/eks)
![PyPI](https://img.shields.io/pypi/v/ensemble-kalman-smoother)
![PyPI Downloads](https://static.pepy.tech/badge/ensemble-kalman-smoother)

This repo contains code to run an Ensemble Kalman Smoother (EKS) for improving pose estimation outputs.

The EKS uses a Kalman smoother to ensemble and smooth pose estimation outputs as a post-processing
step after multiple model predictions have been generated, resulting in a more robust output:

<p align="center"><img src="assets/crim13_singlecam.gif" /></p>

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

## Organizing your data

EKS expects prediction files in **CSV format** using the Lightning Pose / DLC three-row header
(rows: `scorer`, `bodyparts`, `coords`). The ensemble is built from multiple files produced by
different model training runs — referred to here as **seeds** (e.g. `rng=0`, `rng=1`, ...).
EKS uses the variation across seeds to estimate uncertainty and guide smoothing.
At least two seeds are required; three or more are recommended.

### Single-view datasets

For `singlecam` and `ibl-pupil`, place all seed CSV files in a single directory. Every CSV
file found in that directory is treated as one ensemble member; file names can be anything.

```
input_dir/
    predictions.rng=0.csv
    predictions.rng=1.csv
    predictions.rng=2.csv
```

### Multi-view datasets

#### Separate-file format (`multicam`)

Each camera and each seed produces its own CSV file. All files must reside in the same directory.

```
input_dir/
    session_Cam-A_rng=0.csv
    session_Cam-A_rng=1.csv
    session_Cam-B_rng=0.csv
    session_Cam-B_rng=1.csv
    session_Cam-C_rng=0.csv
    session_Cam-C_rng=1.csv
    calibration.toml          # required only for nonlinear EKS (see below)
```

**Camera–file matching**: EKS identifies which file belongs to which camera by checking whether
the camera name is a substring of the filename. A file named `session_Cam-A_rng=0.csv` matches
camera `Cam-A` because `Cam-A` appears in the filename. A few rules follow from this:

- Every camera must appear in at least one filename, and every file in the directory must match
  exactly one camera name. Files that match no camera name are ignored.
- Camera names must not be substrings of one another (e.g. avoid naming cameras `Cam` and
  `Cam-A` together, since `Cam` would match both).
- Use a consistent naming convention across all cameras so that files sort into the same
  seed order for every camera (e.g. always append `_rng=0`, `_rng=1`, ...). This ensures
  that seed 0 from camera A and seed 0 from camera B correspond to the same training run,
  which is required for correct triangulation in the nonlinear (calibrated) path.
- Every camera must have the same number of seed files.

**Without calibration (linear EKS)**: provide `--camera-names` explicitly:

```console
eks multicam --input-dir ./data/mirror-mouse-separate --bodypart-list paw1LH paw2LF \
    --camera-names top bot --make-plot
```

The order you specify determines the ordering of cameras in the output files.

**With calibration (nonlinear EKS)**: camera names are read directly from the `.toml` file,
so `--camera-names` is not required (and will be ignored if provided). The camera order in the
TOML defines which files map to which camera, so file names must contain the camera names as
they appear in the TOML:

```console
eks multicam --input-dir ./data/fly --bodypart-list L1A L1B \
    --calibration ./data/fly/calibration.toml --make-plot
```

#### Mirrored format (`mirrored-multicam`)

All camera views are stored in a single CSV per seed. Bodypart columns are named
`{bodypart}_{camera}` — for example, cameras `top` and `bot` with bodypart `paw1LH` produce
columns `paw1LH_top` and `paw1LH_bot`. Each seed contributes one such file.

```
input_dir/
    session.rng=0.csv    # columns include: paw1LH_top, paw1LH_bot, ...
    session.rng=1.csv
    session.rng=2.csv
```

Provide `--camera-names` so EKS knows which column suffixes to extract:

```console
eks mirrored-multicam --input-dir ./data/mirror-mouse --bodypart-list paw1LH paw2LF \
    --camera-names top bot --make-plot
```

#### IBL paw (`ibl-paw`)

This subcommand is purpose-built for IBL's asynchronous left/right paw recordings.
Each seed produces two CSV files (one per camera), and per-camera timestamp arrays (`.npy`)
are required for temporal alignment:

```
input_dir/
    session.left.rng=0.csv
    session.left.rng=1.csv
    session.right.rng=0.csv
    session.right.rng=1.csv
    session.timestamps.left.npy
    session.timestamps.right.npy
```

Files are assigned to cameras by substring: any filename containing `left` (but not
`timestamps`) goes to the left camera; `right` goes to the right camera. Timestamp files
must contain `timestamps` and either `left` or `right` in the filename. The number of seed
files must match across cameras.

---

### Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup,
linting, and pull request guidelines.

### Authors

* [Cole Hurwitz](https://github.com/colehurwitz)
* [Keemin Lee](https://github.com/keeminlee)
* [Amol Pasarkar](https://github.com/apasarkar)
* [Matt Whiteway](https://github.com/themattinthehatt)
* [Spirit of claude]
