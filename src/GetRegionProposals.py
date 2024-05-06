from BackboneNetwork        import BackboneNetwork
from ClassConstants         import ClassConstants
from DataManager            import DataManager
from FlirDataset            import FlirDataset
from PathConstants          import PathConstants
from RegionProposalNetwork  import *

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
import cv2

import os

# Determine the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# Flags to switch between built-in and custom implementations
use_built_in_rpn = False

# Create Data Manager Object
# Will download dataset if not already available
data_manager = DataManager()
data_manager.download_datasets()

# Create Path Constants Singleton
data_dir = data_manager.get_download_dir()
PathConstants(data_dir)

# Create input dataset
dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images=10, device=device)

# Create backbone network
backbone = BackboneNetwork()
backbone.to(device)

# Create RPN dataset
rpn_dataset = create_rpn_dataset(backbone, dataset, use_built_in_rpn, device)

# Get sizes of images and feature maps
image_size = dataset[0][0].shape
if use_built_in_rpn:
    feature_map_size = rpn_dataset[0][1].shape
else:
    feature_map_size = rpn_dataset[0][0].shape

# Create RPN
torch.manual_seed(0)
rpn = create_region_proposal_network(image_size, feature_map_size, use_built_in_rpn)
rpn.to(device)

# Load User Weights
curr_file_path = os.path.dirname(__file__)
weights_path = os.path.join(curr_file_path,'weights','rpn_weights.pth')
state_dict = torch.load(weights_path, map_location=device)
rpn.load_state_dict(state_dict['model_state'])
rpn.eval()

# Run single image through data loader
collate_fn = rpn_collate_fn(use_built_in_rpn)
data_loader = DataLoader(rpn_dataset, collate_fn=collate_fn)
for data in data_loader:
    boxes = rpn(*data)[0][0]
    break

# Display detections
img = dataset[0][0]
img_data_all = np.uint8(img.permute(1, 2, 0).numpy())
boxes = boxes[:128].numpy()
plt.imshow(img_data_all)
for idx, gt_box in enumerate(boxes):
    x = gt_box[0]
    y = gt_box[1]
    w = gt_box[2] - x
    h = gt_box[3] - y
    rect = patches.Rectangle((x, y), w, h, color='red', linewidth=3, fill=False)
    plt.gca().add_patch(rect)
plt.show()