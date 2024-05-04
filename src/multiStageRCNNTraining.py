import torch
import os
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from BackboneNetwork        import BackboneNetwork
from ClassConstants         import ClassConstants
from CustomDataset          import CustomDataset
from DataManager            import DataManager
from FlirDataset            import FlirDataset
from NetworkTrainer         import NetworkTrainer
from PathConstants          import PathConstants
from RegionProposalNetwork  import *
from networkHead            import *

from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign

import torchvision.models.detection.rpn as torch_rpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

# Flags to switch between built-in and custom implementations
use_built_in_rpn = False
use_built_in_roi_heads = True

# Create Data Manager Object
# Will download dataset if not already available
data_manager = DataManager()
data_manager.download_datasets()

# Create Path Constants Singleton
data_dir = data_manager.get_download_dir()
PathConstants(data_dir)

# Create input dataset
dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images = 10)

# Create backbone network
backbone = BackboneNetwork()

# Create RPN dataset
rpn_dataset = create_rpn_dataset(backbone, dataset, use_built_in_rpn)

# Get sizes of images and feature maps
image_size = dataset[0][0].shape
if use_built_in_rpn:
    feature_map_size = rpn_dataset[0][1].shape
else:
    feature_map_size = rpn_dataset[0][0].shape

# Create RPN
torch.manual_seed(0)
rpn = create_region_proposal_network(image_size, feature_map_size, use_built_in_rpn)
rpn.train()

# Train the RPN network
optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-3)

collate_fn = rpn_collate_fn(use_built_in_rpn)

run_folder = 'region_proposal_network'

network_trainer = NetworkTrainer(
    data       = rpn_dataset,
    model      = rpn,
    optimizer  = optimizer,
    run_folder = run_folder,
    num_epochs = 1,
    batch_size = 1,
    loss_fn    = rpn_loss_fn,
    collate_fn = collate_fn)

network_trainer.train() 

# Create ROI Heads dataset
rpn.eval()
roi_dataset = create_roi_dataset(rpn, dataset, rpn_dataset, use_built_in_roi_heads)

# Create ROI Heads network
torch.manual_seed(0)
roi_heads = create_roi_heads_network(feature_map_size, use_built_in_roi_heads)
roi_heads.train()

# Train ROI Heads Network
optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-3)

loss_fn = roi_loss_fn(use_built_in_roi_heads)
collate_fn = roi_collate_fn(use_built_in_roi_heads)

run_folder = 'roi_heads_network'

network_trainer = NetworkTrainer(
    data       = roi_dataset,
    model      = roi_heads,
    optimizer  = optimizer,
    run_folder = run_folder,
    num_epochs = 1,
    batch_size = 1,
    loss_fn    = loss_fn,
    collate_fn = collate_fn)

network_trainer.train()

# Get detections for first image
roi_heads.eval()
data_loader = DataLoader(roi_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

for data in data_loader:
    detections = roi_heads(*data)[0]
    break

boxes = detections[0]['boxes']
labels = detections[0]['labels']
scores = detections[0]['scores']

# Filter detections
boxes = boxes[scores > 0.05]
labels = labels[scores > 0.05]

# Display detections
img = dataset[0][0]
img = np.uint8(img.permute(1,2,0))
boxes = boxes.detach().numpy()
plt.imshow(img)
for box, label in zip(boxes, labels):
    x = np.round(box[0])
    y = np.round(box[1])
    w = np.round(box[2]) - x + 1
    h = np.round(box[3]) - y + 1
    rect = patches.Rectangle((x, y), w, h, color='red', linewidth=3, fill=False)
    plt.text(x,y,list(ClassConstants.LABELS.keys())[label],color='white',bbox=dict(facecolor='red', edgecolor='red', boxstyle="Square, pad=0"))
    plt.gca().add_patch(rect)
plt.show()