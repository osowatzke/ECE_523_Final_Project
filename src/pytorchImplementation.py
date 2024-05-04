import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from FlirDataset import FlirDataset
from PathConstants import PathConstants
from torch.utils.data.sampler import Sampler
from BackboneNetwork import BackboneNetwork
import numpy as np
from ClassConstants import ClassConstants
import math
from createClassifierNetwork import classifierNet
from torchvision.ops import MultiScaleRoIAlign
from typing import Dict, List, Optional, Tuple, Union
import warnings
from collections import OrderedDict


device = torch.device('cuda')
torch.cuda.set_device(0)

IMG_HEIGHT = 7
IMG_WIDTH = 7
IMG_CHANNELS = 2048


# Pytorch faster RCNN classifier and bounding box regressor classes
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads



class pytROIHeads(classifierNet):
    
    def __init__(self, num_images_in,
                box_score_thresh=0.05,
                box_nms_thresh=0.5,
                box_detections_per_img=100,
                box_fg_iou_thresh=0.5,
                box_bg_iou_thresh=0.5,
                box_batch_size_per_image=512,
                box_positive_fraction=0.25,
                bbox_reg_weights=None,
                num_classes=None,
                out_channels = 2048,
                training_bool = True
                 ):
        
        super().__init__(num_images_in)

        # Create classifier and bbox regressor networks
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024

        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, num_classes)
        box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        # Training flag
        self.training = training_bool

        # Backbone
        self.backbone = BackboneNetwork()

        # ROI Heads
        self.roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        # Loss function

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Union[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses

        return detections
    

    def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Computes the loss for Faster R-CNN.

        Args:
            class_logits (Tensor)
            box_regression (Tensor)
            labels (list[BoxList])
            regression_targets (Tensor)

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        classification_loss = F.cross_entropy(class_logits, labels)

        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss


    def forward( self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]):
    )
       
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        losses = {}
        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        
       
        

    def trainOneEpoch(self, epoch_index, tb_writer, backbone=None):

        # Loss metrics
        running_loss = 0.
        last_loss = 0.

        # Specify optimizer
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, eps=1e-08, weight_decay=5e-3)

        # For each batch
        for i, data in enumerate(self.training_loader):
           
            # Load data
            inputs, labels = data

            # Run data through backbone
            if backbone:
                imgs = torch.stack((*inputs,))
                # features = torch.tensor(backbone(imgs))
                features = (backbone(imgs))

            # Number of ROIs (removing ROIs with class -1)
            numROIs = 0
            for j in range(len(labels)):
                for p in range(len(labels[j]["labels"])):
                    numROIs = numROIs + int(labels[j]["labels"][p] != -1.)

            # Hold all data
            labels_ = torch.zeros([numROIs, len(self.labels)]).to(device)
            labels_db = np.empty((len(labels), len(self.labels))) # DEBUG
            labels_db[:,3] = 1
            classes_ = torch.zeros([numROIs]).to(device)
            imgs_ = torch.zeros([numROIs, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH]).to(device)
            boxes_ = torch.zeros([numROIs, 4]).to(device)
            boxes_db = torch.zeros([numROIs, 5]).to(device)

            # ROI Pooling
            # For each image in the batch
            totalROIs = 0
            k = 0
            for label in labels:

                # Number of ROIs in this image
                numROIs_this_img = 0
                for p in range(len(label["labels"])):
                    numROIs_this_img = numROIs_this_img + int(label["labels"][p] != -1.)
            
                # For each ROI in image
                j = 0
                numTrueROIs = 0
                temp = torch.zeros(numROIs_this_img, 4).to(device)
                for j in range(len(label["labels"])):

                    # Determine class
                    class_ = label["labels"][j]

                    # If it's not class -1:
                    if class_ != torch.tensor(-1.).to(device):

                        # Creating ideal output
                        labels_[totalROIs, int(class_)] = 1.0
                        classes_[totalROIs] = int(class_)

                        # Extracting bounding boxes
                        boxes_[totalROIs, :] = label["boxes"][j]
                        temp[numTrueROIs,:] = label["boxes"][j]
                        boxes_db[totalROIs] = torch.cat((torch.tensor([k]).to(device), (label["boxes"][j]).to(device)), 0)

                        numTrueROIs = numTrueROIs + 1
                        totalROIs = totalROIs + 1
                k = k + 1
            
            # Experimental: Pytorch RoI Pooling
            imgs_pyt = torchvision.ops.roi_pool(features.to(device), boxes_db.to(device), (7, 7), 0.03125)

            # Re-format feature space to make it compatible with FCL dimensions
            features = torch.flatten(imgs_pyt, start_dim = 1)

            # Zero gradients before each batch
            optimizer.zero_grad()

            # Run batch through network
            outputs = self(features)

            # Calculate loss and gradients
            classes_ = classes_.type(torch.LongTensor).to(device)
            loss = self.loss_function(outputs, classes_)
            loss.backward()

            # Clip gradient
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            
            # Adjust weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 10 == 0:
                last_loss = running_loss / 10
                print(' batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * self.numSamples + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0
            
        return last_loss


        
