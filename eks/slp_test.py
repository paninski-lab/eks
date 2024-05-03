from sleap_io.io.slp import read_labels
import os

# python scripts/singlecam_example.py --input-dir ./data/fish-slp --data-type slp --bodypart-list chin mouth head middle tail

base_dir = "data/fish-slp/"
filenames = [
    "4fish.v009.slp.240422_114719.predictions.slp"

]

'''
    "4fish.v009.slp.240422_114719.predictions.slp"
    "4fish.v009.slp.240422_154713.predictions.slp"
    "4fish.v009.slp.240422_154713.predictions.slp",
    "4fish.v009.slp.240422_182825.predictions.slp",
    "4fish.v009.slp.240423_113502.predictions.slp",
    "4fish.v009.slp.240423_141211.predictions.slp",
'''

for f, filename in enumerate(filenames):
    filepath = os.path.join(base_dir, filename)
    labels = read_labels(filepath)
    print(labels[16][1][4].x)
# labels.labeled_frames[frame].instances[animal#][bodypart#].x)

    '''
    nodes = labels.skeletons[0].nodes
    print(nodes)
    keypoint_names = []
    for node in enumerate(nodes):
        keypoint_name = node[1].name
        keypoint_names.append(keypoint_name)
    print(keypoint_names)
    '''
    # labeled_frame = labels[0]
    # instance = labeled_frame[0]
    # print(f'Instance 1: {instance}')
    # print(instance[0])