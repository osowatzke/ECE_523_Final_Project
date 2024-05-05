import torch
import torch.nn as nn
import torchvision.models.detection.rpn as torch_rpn

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList

from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign

from BackboneNetwork import BackboneNetwork
from ClassConstants  import ClassConstants
from RegionProposalNetwork import *
from networkHead import*

def rcnn_collate_fn(data):
    images = []
    targets = []
    for sample in data:
        images.append(sample[0])
        targets.append(sample[1])
    images = torch.cat(images)
    images = images.reshape((len(data),) + data[0][0].shape)
    return images, targets

def rcnn_loss_fn(model_output):
    loss_dict = model_output[1]
    losses = sum(loss for loss in loss_dict.values())
    return losses

class FasterRCNN(nn.Module):

    def __init__(self, image_size, use_built_in_rpn=False, use_built_in_roi_heads=True):
        super().__init__()
        self.image_size = image_size
        self.use_built_in_rpn = use_built_in_rpn
        self.use_built_in_roi_heads = use_built_in_roi_heads
        self.backbone = BackboneNetwork()
        self.__get_feature_map_size()
        self.__create_rpn() 
        self.__create_roi_heads()
    
    def __get_feature_map_size(self):
        input = torch.zeros((1,) + self.image_size)
        output = self.backbone(input)
        self.feature_map_size = output.shape

    def __create_rpn(self):
        self.rpn = create_region_proposal_network(
            image_size       = self.image_size,
            feature_map_size = self.feature_map_size,
            use_built_in_rpn = self.use_built_in_rpn)

    def __create_roi_heads(self):
        self.roi_heads = create_roi_heads_network(
            feature_map_size       = self.feature_map_size,
            use_built_in_roi_heads = self.use_built_in_roi_heads)
        
    def to(self,device):
        super().to(device)
        self.rpn.to(device)

    def forward(self, images, targets=None):
        feature_maps = self.backbone(images)
        image_sizes = (self.image_size[-2:],) * images.shape[0]
        image_list = ImageList(images, image_sizes)
        if self.use_built_in_rpn:
            feature_maps = {'0': feature_maps}
            proposals, rpn_loss = self.rpn(image_list, feature_maps, targets)
        else:
            target_boxes = [target['boxes'] for target in targets]
            proposals, rpn_loss = self.rpn(feature_maps, target_boxes)
        if not isinstance(feature_maps,dict):
            feature_maps = {'0': feature_maps}
        detections, detection_loss = self.roi_heads(feature_maps, proposals, image_sizes, targets)
        losses = {}
        losses.update(rpn_loss)
        losses.update(detection_loss)
        return detections, losses

if __name__ == "__main__":
    
    from DataManager    import DataManager
    from FlirDataset    import FlirDataset
    from NetworkTrainer import NetworkTrainer
    from PathConstants  import PathConstants

    import numpy as np
    import random

    # Create path constants singleton
    data_manager = DataManager()
    data_manager.download_datasets()
    data_dir = data_manager.get_download_dir()
    PathConstants(data_dir)

    # Set the initial random number generator seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Determine the device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Create dataset object
    train_data = FlirDataset(PathConstants.TRAIN_DIR, device=device)

    # Create Faster RCNN Network
    model = FasterRCNN(train_data[0][0].shape)
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    
    # Set the period for saving data
    # -1 will cause data not to be saved
    save_period = {'epoch' : 1, 'batch' : -1}
    
    # Run subfolder
    run_folder = 'custom_faster_rcnn'

    # Create network trainer
    net_trainer = NetworkTrainer(
        data        = train_data, 
        model       = model,
        optimizer   = optimizer,
        run_folder  = run_folder,
        num_epochs  = 50,
        batch_size  = 50,
        loss_fn     = rcnn_loss_fn,
        collate_fn  = rcnn_collate_fn,
        save_period = save_period,
        device      = device
    )

    net_trainer.train()