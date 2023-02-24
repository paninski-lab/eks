# eks
Ensembling and kalman smoothing for pose estimation

### Pupil
The ```ensemble_kalman_pupil_example.py``` script requires a ```model-dir``` which contains lightning-pose or DLC model predictions. To run this script you can execute the following command from a local ```eks``` folder:

```console 
python scripts/ensemble_kalman_pupil_example.py -model-dir <PATH-TO-MODEL-DIR>
```

### Multiview Paw
The ```ensemble_kalman_multiview_paw_example.py``` script requires a ```model-dir``` which contains lightning-pose or DLC model predictions for the left and right camera views. Also, this repository needs timestamp files to align the two cameras. To run this script you can execute the following command:

```console 
python scripts/ensemble_kalman_multiview_paw_example.py -model-dir <PATH-TO-MODEL-DIR>
```
