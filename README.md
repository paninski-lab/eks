# eks
Ensembling and kalman smoothing for pose estimation

### Pupil
The ```ensemble_kalman_pupil_example.py``` script requires a ```model-dir``` which contains lightning-pose or DLC model predictions. To run this script you can execute the following command from a local ```eks``` folder:

```console 
python scripts/ensemble_kalman_pupil_example.py -model-dir <PATH-TO-MODEL-DIR>
```

### Multiview Paw
The ```ensemble_kalman_multiview_paw_example.py``` script requires a ```model-dir``` which contains lightning-pose or DLC model predictions for the left and right camera views. Also, this director needs timestamp files to align the two cameras. To run this script you can execute the following command:

```console 
python scripts/ensemble_kalman_multiview_paw_example.py -model-dir <PATH-TO-MODEL-DIR>
```

### Multicam
The ```ensemble_kalman_multicam_example.py``` script requires a ```parquet-path``` which contains lightning-pose predictions. You also need to specify the video name to be processed, the number of train frames and model type, the keypoints to be processed, and the camera names. To run this script for the mirror-mouse dataset, you can execute the following command:

```console 
python scripts/ensemble_kalman_multicam_example.py -parquet-path <PATH-TO-PARQUET-FILE> -video-name <VIDEO-NAME> -train-frames 1 -model-type 'semi-super context' -keypoint-ensemble-list 'paw1LH' 'paw2LF' -camera-names 'top' 'bot’
```