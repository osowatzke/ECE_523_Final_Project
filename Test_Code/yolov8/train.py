#####################################################################
# Authors: Nick Blanchard           | Owen Sowatzke
#          nickyblanch@arizona.edu  |
#
# Code to train an Ultralytics YOLOv8 model on the ADAS Dataset V2
#####################################################################


# DEPDENCIES ########################################################

from ultralytics import YOLO


# TRAINING ##########################################################

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='flir_adas_v2.yaml', epochs=100, imgsz=640)


# VALIDATION #########################################################

# Evaluate model performance on the validation set
metrics = model.val()
