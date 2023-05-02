# EKS
This repo contains code to run an Ensemble Kalman Smoother (EKS) for improving pose estimation outputs. 

## Installation

First you'll have to install the `git` package in order to access the code on github. 
Follow the directions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
for your specific OS.
Then, in the command line, navigate to where you'd like to install the `eks` package and move 
into that directory:
```
$: git clone https://github.com/colehurwitz/eks
$: cd eks
```

Install the requirements:
```
$: pip install -r requirements.txt 
```

To make the package modules visible to the python interpreter, locally run pip 
install from inside the main `eks` directory:

```
$: pip install -e .
```

## Example scripts

We provide several example datasets and fitting scripts to illustrate use of the package.

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
python scripts/multicam_example.py --csv-dir ./data/mirror-mouse --bodypart-list paw1LH paw2LF paw3RF paw4RH --camera-names top bot
```

### IBL pupil dataset
The `pupil_example.py` script requires a `csv-dir` which contains lightning-pose or DLC 
model predictions. 
To run this script on the example data provided, execute the following command from inside this repo:

```console 
python scripts/pupil_example.py --csv-dir ./data/ibl-pupil
```

### IBL paw dataset (multiple asynchronous views)
The `multiview_paw_example.py` script requires a `csv-dir` which contains lightning-pose 
or DLC model predictions for the left and right camera views, as well as timestamp files to align 
the two cameras. 
To run this script on the example data provided, execute the following command from inside this repo:

```console 
python scripts/multiview_paw_example.py --csv-dir ./data/ibl-paw
```

Authors: Cole Hurwitz and Matt Whiteway
