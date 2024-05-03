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
from RegionProposalNetwork  import RegionProposalNetwork

from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign

def collate_fn(data):
    images = []
    targets = []
    for sample in data:
        images.append(sample[0])
        targets.append(sample[1])
    images = torch.cat(images)
    return images, targets                     

def roi_collate_fn(data):
    features    = []
    proposals   = []
    image_sizes = []
    targets     = []
    for sample in data:
        features.append(sample[0])
        proposals.append(sample[1])
        image_sizes.append(sample[2])
        targets.append(sample[3])
    features = {'0' : torch.cat(features)}
    return features, proposals, image_sizes, targets

run_backbone = True

# Create Data Manager Object
# Will download dataset if not already available
data_manager = DataManager()
data_manager.download_datasets()

# Create Path Constants Singleton
data_dir = data_manager.get_download_dir()
PathConstants(data_dir)

# Create Dataset Object
dataset = FlirDataset(PathConstants.TRAIN_DIR) #, num_images = 10)
rpn_dataset = CustomDataset()

# Run images through backbone network
if run_backbone:
    targets = []
    feature_maps = []
    backbone_network = BackboneNetwork()
    for idx in range(len(dataset)):
        img = dataset[idx][0]
        tgt = dataset[idx][1]['boxes']
        img = img.reshape((1,) + img.shape)
        feature_map = backbone_network(img)
        rpn_dataset.append((feature_map, tgt))

torch.manual_seed(0)

rpn = RegionProposalNetwork(dataset[0][0].shape, rpn_dataset[0][0].shape)

if True:
    optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-3)

    network_trainer = NetworkTrainer(
        data       = rpn_dataset,
        model      = rpn,
        optimizer  = optimizer,
        num_epochs = 10,
        batch_size = 8,
        collate_fn = collate_fn)

    network_trainer.train()

file_name = os.path.dirname(__file__)
file_name = os.path.join(file_name, 'run', 'run__2024-05-02_23-08-34','cp__epoch_9_batch_0.pth')
state_dict = torch.load(file_name)
rpn.load_state_dict(state_dict['model_state'])
rpn.eval()

roi_dataset = CustomDataset([])
print(len(roi_dataset))

for idx in range(len(rpn_dataset)):
    feature_map = rpn_dataset[idx][0]
    tgt = dataset[idx][1]
    image_sizes = dataset[idx][0].shape[-2:]
    proposals = rpn(feature_map)[0]
    # print(feature_map)
    # print(proposals)
    # print(image_sizes)
    # print(tgt)
    roi_dataset.append((feature_map, proposals, image_sizes, tgt))
    #print(roi_dataset[idx])
    #print(len(roi_dataset[idx]))

box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

resolution = box_roi_pool.output_size[0]
representation_size = 1024
out_channels = rpn_dataset[0][0].shape[1]
box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

num_classes = len(ClassConstants.LABELS.keys())
representation_size = 1024
box_predictor = FastRCNNPredictor(representation_size, num_classes)

bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)

roi_heads = RoIHeads(
    box_roi_pool         = box_roi_pool,
    box_head             = box_head,
    box_predictor        = box_predictor,
    # Faster R-CNN training
    fg_iou_thresh        = 0.5,
    bg_iou_thresh        = 0.5,
    batch_size_per_image = 512,
    positive_fraction    = 0.25,
    bbox_reg_weights     = bbox_reg_weights,
    # Faster R-CNN inference
    score_thresh         = 0.05,
    nms_thresh           = 0.5,
    detections_per_img   = 100)

'''
# print(*roi_dataset[0])
print(roi_dataset[0][1])
print(roi_dataset[0][2])
print(roi_dataset[0][3])
roi_heads({'0':roi_dataset[0][0]},roi_dataset[0][1],[roi_dataset[0][2]],[roi_dataset[0][3]])
'''

if True:
    optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-3)

    network_trainer = NetworkTrainer(
        data       = roi_dataset,
        model      = roi_heads,
        optimizer  = optimizer,
        num_epochs = 10,
        batch_size = 8,
        collate_fn = roi_collate_fn)

    network_trainer.train()
        
feature_map = rpn_dataset[0][0]
boxes = rpn(feature_map)[0]
boxes = boxes.detach().numpy()
img = dataset[0][0]
img = np.uint8(img.permute(1,2,0))
plt.imshow(img)
for idx, box in enumerate(boxes):
    x = np.round(box[0])
    y = np.round(box[1])
    w = np.round(box[2]) - x + 1
    h = np.round(box[3]) - y + 1
    rect = patches.Rectangle((x, y), w, h, color='red', linewidth=3, fill=False)
    plt.gca().add_patch(rect)
plt.show()





