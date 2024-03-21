#####################################################################
# Authors: Nick Blanchard           | Owen Sowatzke
#          nickyblanch@arizona.edu  |
#
# Code to train an Ultralytics YOLOv8 model on the ADAS Dataset V2
#####################################################################


# DEPDENCIES ########################################################

from ultralytics import YOLO
from json_parser import JsonParser
import os
import shutil

# COPY DATA #########################################################
path_this_file = os.path.split(os.path.abspath(__file__))[0]
run_dir = os.path.join(path_this_file,'..','..','run')
datasets_dir = os.path.join(run_dir,'datasets')
data_dir = os.path.join(path_this_file,'..','..','FLIR_ADAS_v2')
training_dir = os.path.join(data_dir, 'images_thermal_train','data')
if not os.path.isdir(os.path.join(datasets_dir, 'training')):
    os.makedirs(os.path.join(datasets_dir, 'training'))
    shutil.copytree(training_dir, os.path.join(datasets_dir, 'training', 'images'))
parser = JsonParser(os.path.join(training_dir,'..','index.json'),  os.path.join(datasets_dir, 'training', 'labels'))
parser.run()
training_dir = os.path.join(data_dir, 'video_thermal_test','data')
if not os.path.isdir(os.path.join(datasets_dir, 'test')):
    os.makedirs(os.path.join(datasets_dir, 'test'))
    shutil.copytree(training_dir, os.path.join(datasets_dir, 'test', 'images'))
parser = JsonParser(os.path.join(training_dir,'..','index.json'),  os.path.join(datasets_dir, 'test', 'labels'))
parser.run()
training_dir = os.path.join(data_dir, 'images_thermal_val','data')
if not os.path.isdir(os.path.join(datasets_dir, 'valid')):
    os.makedirs(os.path.join(datasets_dir, 'valid'))
    shutil.copytree(training_dir, os.path.join(datasets_dir, 'valid', 'images'))
parser = JsonParser(os.path.join(training_dir,'..','index.json'),  os.path.join(datasets_dir, 'valid', 'labels'))
parser.run()

# TRAINING ##########################################################

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='flir_adas_v2.yaml', epochs=100, imgsz=640)


# VALIDATION #########################################################

# Evaluate model performance on the validation set
metrics = model.val()
