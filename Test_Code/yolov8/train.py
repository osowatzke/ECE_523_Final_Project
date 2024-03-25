#####################################################################
# Authors: Nick Blanchard           | Owen Sowatzke
#          nickyblanch@arizona.edu  |
#
# Code to train an Ultralytics YOLOv8 model on the ADAS Dataset V2
#####################################################################

SETUP_DATASET = False
TRAIN = False
TEST = True

# DEPENDENCIES ########################################################

from ultralytics import YOLO
from json_parser import JsonParser
import os
import shutil
import glob
import random


# SETUP DATASET #######################################################

if SETUP_DATASET:
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

    force_flag = True
    setup_dir('images_thermal_train', force_flag)
    setup_dir('images_thermal_val', force_flag)
    setup_dir('video_thermal_test', force_flag)


# TRAINING ##########################################################

if TRAIN:

    # Load a model
    model = YOLO('yolov8n.yaml')

    # Train the model
    results = model.train(data='./Test_Code/yolov8/flir_adas_v2.yaml', epochs=1, imgsz=640)

    # Evaluate model performance on the validation set
    metrics = model.val()

if TEST:

    # Load model
    model = YOLO('./runs/detect/train7/weights/best.pt')  # load a custom model

    # Select 10 random images
    path = os.getcwd()
    imgs = glob.glob(os.path.join(path, "./run/datasets/test/images/*.jpg"))
    random_nums = []
    for i in range(0, 20):
        # Test inference
        test = model(imgs[random.randint(1, len(imgs))], show=True)  # predict on an image

