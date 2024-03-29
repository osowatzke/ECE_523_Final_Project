#####################################################################
# Authors: Nick Blanchard           | Owen Sowatzke
#          nickyblanch@arizona.edu  |
#
# Code to train an Ultralytics YOLOv8 model on the ADAS Dataset V2
#####################################################################


# DEPDENCIES ########################################################

from ultralytics import YOLO
from json_parser import JsonParser
from class_labels import CLASS_LABELS
import os
import shutil

# COPY DATA #########################################################
def setup_dir(sub_dir, force=False):
    path_this_file = os.path.split(os.path.abspath(__file__))[0]
    in_dir = os.path.join(path_this_file,'..','..','FLIR_ADAS_v2',sub_dir)
    out_dir = os.path.join(path_this_file,'..','..','run','datasets')
    out_dir = os.path.join(out_dir, sub_dir.split('_')[-1])
    dir_exists = os.path.isdir(out_dir)
    create_dir = not dir_exists or force
    if create_dir:
        if dir_exists:
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        in_image_dir = os.path.join(in_dir, 'data')
        out_image_dir = os.path.join(out_dir, 'images')
        shutil.copytree(in_image_dir, out_image_dir)
        in_json_file = os.path.join(in_dir, 'index.json')
        label_dir = os.path.join(out_dir, 'labels')
        parser = JsonParser(in_json_file, label_dir)
        parser.run()

force_flag = False
setup_dir('images_thermal_train', force_flag)
setup_dir('images_thermal_val', force_flag)
setup_dir('video_thermal_test', force_flag)

path_this_file = os.path.split(os.path.abspath(__file__))[0]
run_dir = os.path.join(path_this_file,'..','..','run')
dataset_dir = os.path.join(run_dir, 'datasets')
yaml_path = os.path.join(run_dir, 'train.yaml')
with open(yaml_path, "w") as file:
    file.write(f'path: {os.path.abspath(dataset_dir)}\n')
    file.write('train: train\n')
    file.write('val: val\n')
    file.write('test: test\n')
    file.write('\n')
    file.write('names:\n')
    for (key, val) in zip(CLASS_LABELS.keys(), CLASS_LABELS.values()):
        file.write(f'  {val}: {key}\n')

# TRAINING ##########################################################

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data=yaml_path, epochs=100, imgsz=640)


# VALIDATION #########################################################

# Evaluate model performance on the validation set
metrics = model.val()
