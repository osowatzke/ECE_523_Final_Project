import torch
import torch.nn              as nn
import torch.nn.functional   as F
import torchvision.ops       as ops
import torchvision.ops.boxes as box_ops

import torchvision.models.detection.rpn as torch_rpn
from   torch.utils.data import DataLoader
from   torchvision.models.detection.anchor_utils import AnchorGenerator
from   torchvision.models.detection.image_list import ImageList

from BackboneNetwork import *
from CustomDataset   import CustomDataset

import AnchorBoxUtilities   as anchor_utils
import BoundingBoxUtilities as bbox_utils
import SamplingUtilities    as sample_utils

class RegionProposalNetwork(nn.Module):

    class ConvLayers(nn.Module):
    
        def __init__(self, in_channels,hidden_layer_channels=512,num_anchors=9):
            super().__init__()

            # Shared convolution layer
            self.conv1 = nn.Conv2d(in_channels,hidden_layer_channels,3,padding=1)

            # Convolution layer for classifier (produces A x H x W output)
            self.conv2 = nn.Conv2d(hidden_layer_channels,num_anchors,1)

            # Convolution layer for region proposals (produces 4*A x H x W output)
            self.conv3 = nn.Conv2d(hidden_layer_channels,4*num_anchors,1)

            for layer in self.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

        def forward(self, feature_maps):
            shared_conv = F.relu(self.conv1(feature_maps))
            cls_pred = self.conv2(shared_conv)
            bbox_offsets = self.conv3(shared_conv)
            return cls_pred, bbox_offsets

    # Class constructor
    def __init__(
        self, 
        image_size,
        feature_map_size,
        anchor_box_sizes      = (32,64,128),
        aspect_ratios         = (0.5,1.0,2.0),
        min_proposal_score    = 0.0,
        nms_thresh            = 0.7,
        fg_iou_thresh         = 0.7,
        bg_iou_thresh         = 0.3,
        max_samp_pre_nms      = 1024,
        max_samp_post_nms     = 512,
        batch_size            = 256,
        pos_frac              = 0.5):

        super().__init__()

        # Save input arguments as member variables
        self.image_size         = image_size
        self.min_proposal_score = min_proposal_score
        self.nms_thresh         = nms_thresh
        self.fg_iou_thresh      = fg_iou_thresh
        self.bg_iou_thresh      = bg_iou_thresh
        self.max_samp_pre_nms   = max_samp_pre_nms
        self.max_samp_post_nms  = max_samp_post_nms
        self.batch_size         = batch_size
        self.pos_frac           = pos_frac
        self.min_size           = 1e-3

        # Generate anchor boxes (same for all images)
        self.anchor_boxes = anchor_utils.gen_anchor_boxes(
            image_size,
            feature_map_size, 
            anchor_box_sizes, 
            aspect_ratios)

        # Get dimensions for neural network layers
        in_channels = feature_map_size[-3]
        self.num_anchors = len(anchor_box_sizes) * len(aspect_ratios)

        # Create convolutional neural network layers
        self.conv_layers = self.ConvLayers(in_channels, in_channels, self.num_anchors)

    def to(self,device):
        super().to(device)
        self.anchor_boxes = self.anchor_boxes.to(device)

    def forward(self,feature_maps,targets=None):

        # Run convolution layers
        cls_pred, bbox_off_pred = self.conv_layers(feature_maps)

        # Format output of convolutional layers to form N x 1 and N x 4 matrices
        cls_pred      = self.format_conv_output(cls_pred)
        bbox_off_pred = self.format_conv_output(bbox_off_pred)

        # Get bounding box from offsets
        bbox_pred = bbox_utils.centroids_to_corners(bbox_off_pred.detach(), self.anchor_boxes)

        # Select best bounding boxes
        bbox_pred = self.get_best_proposals(bbox_pred, cls_pred)[0]

        # Default value for loss
        losses = {}

        # Only compute loss when training
        if self.training:

            # Get truth data
            cls_truth, bbox_truth = self.get_ground_truth_data(targets, self.fg_iou_thresh, self.bg_iou_thresh)

            # Convert ground truth bounding boxes to ground truth offsets
            bbox_off_truth = bbox_utils.corners_to_centroid(bbox_truth, self.anchor_boxes)
            
            # Function computes the loss
            bbox_loss, cls_loss = self.compute_loss(bbox_off_pred, bbox_off_truth, cls_pred, cls_truth)

            losses = {
                "loss_objectness": cls_loss,
                "loss_rpn_box_reg": bbox_loss,
            }

        return bbox_pred, losses
    
    # Function packs the output of a convolution neural network into an N x M array
    def format_conv_output(self, x):
        x = x.reshape(x.shape[0:2] + (-1,))
        x = x.permute(0,2,1)
        w = x.shape[2] // self.num_anchors
        if w == 1:
            x = x.ravel()
        else:
            x = x.reshape(-1, w)
        return x
    
    def get_ground_truth_data(self, targets, max_iou_thresh, min_iou_thresh):

        # Allocate tensors for truth data
        cls_truth  = torch.zeros((len(targets), self.anchor_boxes.shape[0]), device=self.anchor_boxes.device)
        bbox_truth = torch.zeros((len(targets),) + self.anchor_boxes.shape, device=self.anchor_boxes.device)

        # Populate truth data
        for idx in range(len(targets)):

            # Only change default value if there are targets in image
            if targets[idx].numel() != 0:

                # Get IOU between every target and anchor box
                iou = ops.box_iou(targets[idx], self.anchor_boxes)

                # Determine which ground truth box provides the maximum IOU
                max_val, max_idx = iou.max(dim=0)

                # Select the best ground truth box as truth data
                # The class will be used to mask out invalid values
                bbox_truth[idx,:,:] = targets[idx][max_idx,:]

                # Default the class data to -1 for invalid bounding box
                cls_truth[idx,:] = torch.full((self.anchor_boxes.shape[0],), -1)

                # Determine which anchor boxes are foreground
                fg_idx = torch.where(max_val >= max_iou_thresh)[0]

                # Determine which parts of image are 
                bg_idx = torch.where(max_val < min_iou_thresh)[0]
                
                # Update the true classes of data
                # 1 => foreground
                # 0 => background
                cls_truth[idx,fg_idx] = 1
                cls_truth[idx,bg_idx] = 0

                # Determine which anchor box is maximally overlapped with bounding box
                max_val, max_idx = iou.max(dim=1)

                # THIS IS A BUG!!!!!!!!!
                # max_idx = torch.where(max_val[:,None] == iou)[1]

                # Ensure each object has at least one bounding box
                cls_truth[idx, max_idx] = 1

        # Flatten data
        cls_truth  = cls_truth.ravel()
        bbox_truth = bbox_truth.reshape(-1,4)

        return cls_truth, bbox_truth

    def get_best_proposals(self, bbox_pred, cls_pred):

        # Empty list for boxes and scores for batch
        best_boxes = []
        best_scores = []

        # Compute the total number of anchor boxes
        total_num_anchors = self.anchor_boxes.shape[0]

        cls_pred = cls_pred.detach()

        # Reshape the predictions to easily separate data from each image
        cls_pred = cls_pred.reshape(-1, total_num_anchors) 
        bbox_pred = bbox_pred.reshape(-1, total_num_anchors, 4)
        
        # Determine the number of images in the batch
        num_images = cls_pred.shape[0]

        # Loop for each image
        for img in range(num_images):

            # Extract predictions for current image
            scores, boxes = cls_pred[img], bbox_pred[img]

            # Get N boxes with highest score before non-max suppression
            _, idx = scores.topk(self.max_samp_pre_nms)
            boxes, scores = boxes[idx], scores[idx]

            # Make the score probability-like by passing through a sigmoid function
            scores = torch.sigmoid(scores)

            # Clip bounding boxes to image
            boxes = box_ops.clip_boxes_to_image(boxes, self.image_size[-2:])

            # Remove small boxes
            idx = box_ops.remove_small_boxes(boxes, self.min_size)
            boxes, scores = boxes[idx], scores[idx]

            # Remove low scoring boxes
            idx = torch.where(scores >= self.min_proposal_score)[0]
            boxes, scores = boxes[idx], scores[idx]

            # Perform non-maximum suppression
            idx = box_ops.nms(boxes, scores, self.nms_thresh)

            # Keep only N top scoring predictions
            idx = idx[:self.max_samp_post_nms]
            boxes, scores = boxes[idx], scores[idx]

            # Save boxes and scores from this batch
            best_boxes.append(boxes)
            best_scores.append(scores)

        return best_boxes, best_scores
    
    def compute_loss(self, bbox_off_pred, bbox_off_truth, cls_pred, cls_truth):
        
        # Compute the total number of anchor boxes
        total_num_anchors = self.anchor_boxes.shape[0]

        # Sample positive and negative samples from each image
        pos_samp, neg_samp = sample_utils.sample_data(cls_truth, total_num_anchors, self.batch_size, self.pos_frac)

        # Create tensor with all samples
        all_samp = torch.cat([pos_samp, neg_samp])

        # Determine the bounding box loss
        bbox_loss = F.smooth_l1_loss(
            bbox_off_pred[pos_samp],
            bbox_off_truth[pos_samp],
            beta=1/9,
            reduction="sum",
        ) / (all_samp.numel())

        # Determine the classifier loss
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred[all_samp], cls_truth[all_samp].type(torch.float))

        return bbox_loss, cls_loss


def create_region_proposal_network(image_size, feature_map_size, use_built_in_rpn=False):

    if use_built_in_rpn:

        # Create Anchor Generator
        anchor_box_sizes = ((32,64,128),)
        aspect_ratios    = ((0.5,1.0,2.0),)
        anchor_generator = AnchorGenerator(anchor_box_sizes, aspect_ratios)

        # Create region proposal head
        num_anchors = len(anchor_box_sizes[0]) * len(aspect_ratios[0])
        rpn_head = torch_rpn.RPNHead(feature_map_size[1], num_anchors)

        # Limits for non-max suppression
        pre_nms_top_n  = {'training' : 1024, 'testing' : 1024}
        post_nms_top_n = {'training' : 512,  'testing' : 512}

        # Create built-in region proposal network
        rpn = torch_rpn.RegionProposalNetwork(
            anchor_generator     = anchor_generator,
            head                 = rpn_head,
            fg_iou_thresh        = 0.7,
            bg_iou_thresh        = 0.3,
            batch_size_per_image = 256,
            positive_fraction    = 0.5,
            pre_nms_top_n        = pre_nms_top_n,
            post_nms_top_n       = post_nms_top_n,
            nms_thresh           = 0.7,
            score_thresh         = 0.0)
        
    else:

        # Create user-defined region proposal network
        rpn = RegionProposalNetwork(
            image_size       = image_size,
            feature_map_size = feature_map_size)

    return rpn


def create_rpn_dataset(backbone, dataset, use_built_in_rpn=False):
    rpn_dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=backbone_collate_fn)
    for image, targets in data_loader:
        feature_map = backbone(image)
        if use_built_in_rpn:
            sample = (image, feature_map, targets[0])
        else:
            targets = targets[0]['boxes']
            sample = (feature_map, targets)
        rpn_dataset.append(sample)
    return rpn_dataset


class rpn_collate_fn:
    def __init__(self, use_built_in_rpn=False):
        if use_built_in_rpn:
            self._collate_fn = self._builtin_fn
        else:
            self._collate_fn = self._user_fn

    def __call__(self, data):
        return self._collate_fn(data)
    
    def _user_fn(self, data):
        features = []
        targets  = []
        for sample in data:
            features.append(sample[0])
            targets.append(sample[1])
        features = torch.cat(features)
        return features, targets  

    def _builtin_fn(self, data):
        images   = []
        features = []
        targets  = []
        for sample in data:
            images.append(sample[0])
            features.append(sample[1])
            targets.append(sample[2])
        images      = torch.cat(images)
        num_images  = images.shape[0]
        image_sizes = (images[0].shape[-2:],) * num_images
        images      = ImageList(images, image_sizes)
        features    = {'0' : torch.cat(features)}
        return images, features, targets


def rpn_loss_fn(model_output):
    loss_dict = model_output[1]
    losses = sum(loss for loss in loss_dict.values())
    return losses


if __name__ == "__main__":

    import math

    from DataManager     import DataManager
    from PathConstants   import PathConstants
    from FlirDataset     import FlirDataset
    
    # Hyperparameters
    num_images = 2
    batch_size = 2
    num_epochs = 10

    # Compute the number of batches
    num_batches = math.ceil(num_images/batch_size)

    # Download dataset
    data_manager = DataManager('train')
    data_manager.download_datasets()
    data_dir = data_manager.get_download_dir()
    PathConstants(data_dir)

    # Create dataset object
    dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images=num_images)

    # Create backbone object
    backbone = BackboneNetwork()

    # Create dataset for user-defined RPN
    rpn_dataset = create_rpn_dataset(backbone, dataset, False)

    # Extract image sizes and feature map sizes
    image_size       = dataset[0][0].shape
    feature_map_size = rpn_dataset[0][0].shape

    # Create user-defined region proposal network
    torch.manual_seed(0)
    rpn = create_region_proposal_network(image_size, feature_map_size, False) 
    rpn.train()
    
    # Create data loader
    torch.manual_seed(0)
    collate_fn = rpn_collate_fn(False)
    rpn_data_loader = DataLoader(rpn_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Create optimizer    
    optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-3)

    # Loop for each epoch
    for epoch in range(num_epochs):

        # Loop for each batch
        for args in rpn_data_loader:

            # Run user-defined model
            optimizer.zero_grad()
            model_output = rpn(*args)
            losses = rpn_loss_fn(model_output)
            losses.backward()
            optimizer.step()
    
        # Print training results
        print(epoch, losses.item())

    # Print newline to separate data
    print()
    
    # Create dataset for built-in RPN
    rpn_dataset = create_rpn_dataset(backbone, dataset, True)

    # Create built-in region proposal network
    torch.manual_seed(0)
    rpn = create_region_proposal_network(image_size, feature_map_size, True) 
    rpn.train()
    
    # Create data loader
    torch.manual_seed(0)
    collate_fn = rpn_collate_fn(True)
    rpn_data_loader = DataLoader(rpn_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Create optimizer    
    optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-3)

    # Loop for each epoch
    for epoch in range(num_epochs):
        
        # Loop for each batch
        for args in rpn_data_loader:

            # Train built in network
            optimizer.zero_grad()
            model_output = rpn(*args)
            losses = rpn_loss_fn(model_output)
            losses.backward()
            optimizer.step()

        # Print training results
        print(epoch, losses.item())