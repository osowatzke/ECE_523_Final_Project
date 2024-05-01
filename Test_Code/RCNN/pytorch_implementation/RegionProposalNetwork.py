import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
import torchvision.ops.boxes as box_ops
import torch
from torchvision.models.detection.rpn import RPNHead, concat_box_prediction_layers
from torchvision.models.detection.rpn import det_utils

from AnchorBoxUtilities import gen_anchor_boxes
import BoundingBoxUtilities as bbox_utils
from BackboneNetwork import BackboneNetwork

from PathConstants import PathConstants
from FlirDataset import FlirDataset

# Run RPN head
#   scores
#   boxDeltas

# Generate Anchor Boxes

# Convert boxDeltas into proposals
#   Solve for x, y, w, and h
#     tx = (x - xa)/wa
#     ty = (y - ya)/ha
#     tw = log(w/wa)
#     th = log(h/ha)

# Filter proposals
#   Select top N boxes
#   Convert scores to probabilties
#   Clip boxes to image
#   Remove small boxes
#   Remove low scoring boxes
#   Non-max suppression
#   Keep only top N proposals

# Assign targets to nearest anchors

# Encode regression targets
#   Solve for tx, ty, tw, and th for each proposal

# Compute loss
#   box_loss = smooth_l1_loss (computed on positive indices only)
#   objectness_loss = binary_cross_entropy_width_logits (on all samples)


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

        def forward(self, feature_maps):

            # Create empty list for layer outputs
            cls_pred = [None] * len(feature_maps)
            bbox_offsets = [None] * len(feature_maps)

            # Loop over feature maps for each batch
            for i, feature_map in enumerate(feature_maps):
                shared_conv = F.relu(self.conv1(feature_map))
                cls_pred[i] = F.sigmoid(self.conv2(shared_conv))
                bbox_offsets[i] = self.conv3(shared_conv)

            return cls_pred, bbox_offsets

    # Class constructor
    def __init__(
        self, 
        image_size,
        feature_map_size,
        anchor_box_sizes = (32,64,128),
        aspect_ratios = (0.5,1.0,2.0),
        hidden_layer_channels = 512,
        min_proposal_score = 0.5,
        iou_threshold = 0.7):

        super().__init__()

        # Save input arguments as member variables
        self.image_size = image_size
        self.min_proposal_score = min_proposal_score
        self.iou_threshold = iou_threshold

        # Generate anchor boxes (same for all images)
        self.anchor_boxes = gen_anchor_boxes(
            image_size,
            feature_map_size, 
            anchor_box_sizes, 
            aspect_ratios)
        
        print(self.anchor_boxes.shape)

        # Get dimensions for neural network layers
        in_channels = feature_map_size[-3]
        num_anchors = len(anchor_box_sizes) * len(aspect_ratios)

        # Create convolutional neural network layers
        self.conv_layers = self.ConvLayers(in_channels, hidden_layer_channels, num_anchors)

    def forward(self,feature_maps):

        # Run convolution layers
        cls_pred, bbox_offsets = self.conv_layers(feature_maps)

        # Format output of convolutional layers to form Nx1 and Nx4 matrices
        cls_pred = self.format_cls_pred(cls_pred)
        bbox_offsets = self.format_bbox_offsets(bbox_offsets)

        # Get bounding box from offsets
        bbox_pred = bbox_utils.centroids_to_corners(bbox_offsets, self.anchor_boxes)

        # Reshape to correct dimensions
        cls_pred = cls_pred.reshape(len(feature_maps), -1)
        bbox_pred = bbox_pred.reshape(len(feature_maps), -1, 4)

        # Select best bounding boxes
        bbox_pred, cls_pred = self.get_best_proposals(bbox_pred, cls_pred)

    def format_cls_pred(self, cls_pred):
        cls_pred = torch.stack(cls_pred)
        print(cls_pred.shape)
        cls_pred = cls_pred.reshape(cls_pred.shape[0:3] + (-1,))
        cls_pred = cls_pred.reshape((-1,) + cls_pred.shape[2:])
        cls_pred = cls_pred.permute(0,2,1)
        cls_pred = cls_pred.reshape(-1,1)
        return cls_pred
    
    def format_bbox_offsets(self, bbox_offsets):
        bbox_offsets = torch.stack(bbox_offsets)
        bbox_offsets = bbox_offsets.reshape(bbox_offsets.shape[0:3] + (-1,))
        bbox_offsets = bbox_offsets.reshape((-1,) + bbox_offsets.shape[2:])
        bbox_offsets = bbox_offsets.permute(0,2,1)
        bbox_offsets = bbox_offsets.reshape(-1,4)
        return bbox_offsets
    
    def get_best_proposals(self, bbox_pred, cls_pred):

        pre_nms_top_n = 512
        post_nms_top_n = 128

        best_boxes = []
        best_scores = []

        for boxes, scores in zip(bbox_pred, cls_pred):

            # Get N boxes with highest score before non-max suppression
            _, idx = scores.topk(pre_nms_top_n)
            boxes, scores = boxes[idx], scores[idx]

            # Clip bounding boxes to image
            boxes = box_ops.clip_boxes_to_image(boxes, self.image_size[-2:])

            # Remove small boxes
            idx = box_ops.remove_small_boxes(boxes, 1e-3)
            boxes, scores = boxes[idx], scores[idx]

            # Remove low scoring boxes
            idx = torch.where(scores >= self.min_proposal_score)[0]
            boxes, scores = boxes[idx], scores[idx]

            # Perform non-maximum suppression
            idx = box_ops.nms(boxes, scores, self.iou_threshold)

            # Keep only N top scoring predictions
            idx = idx[:post_nms_top_n]
            boxes, scores = boxes[idx], scores[idx]

            best_boxes.append(boxes)
            best_scores.append(scores)

        return best_boxes, best_scores

    def get_top_n_idx(self, scores, n):
        top_n_idx = []
        n = min(scores.shape[1], n)
        for score in scores.split(scores, 1):
            _, idx = score.topk(n, dim=1)
            top_n_idx.append(idx)
        return top_n_idx



        # (N, C, _, _) = cls_pred.shape
        # cls_pred = cls_pred.reshape(N, C, -1)
        # cls_pred = cls_pred.permute(0, 2, 1)
        # cls_pred = cls_pred.reshape(N, -1, 1)
        # return cls_pred
    
    # def format_bbox_offsets(self, bbox_offsets):
    #     bbox_offsets = torch.stack(bbox_offsets)
    #     (N, C, _, _) = bbox_offsets.shape
    #     bbox_offsets = bbox_offsets.reshape(N, C, -1)
    #     bbox_offsets = bbox_offsets.permute(0, 2, 1)
    #     bbox_offsets = bbox_offsets.reshape(N, -1, 4)
    #     return bbox_offsets

if __name__ == "__main__":
    anchor_generator = AnchorGenerator()
    images = ImageList(torch.zeros(2,3,512,640), [(3,512,640),(3,512,640)])
    anchor_boxes = anchor_generator(images, [torch.zeros(2,512,16,20)])
    #print(len(anchor_boxes))
    
    PathConstants()
    dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images=1)
    backbone = BackboneNetwork()
    img = dataset[0][0].view((1,) + dataset[0][0].shape)
    feature_map = backbone(img)
    # region_prosal_network = RegionProposalNetwork(img.shape, feature_map.shape)
    # print(region_prosal_network.anchor_boxes.shape)
    # region_prosal_network(feature_map)
    #region_proposal_network = RegionProposalNetwork

    head = RPNHead(feature_map.shape[1], 9)
    objectness, pred_bbox_deltas = head.forward([feature_map, feature_map])
    # print(objectness[0].shape)
    # print(pred_bbox_deltas)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    print(num_anchors_per_level_shape_tensors)
    print(num_anchors_per_level)
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    #print(objectness.shape)
    #print(pred_bbox_deltas.shape)
    rpn = RegionProposalNetwork(img.shape,feature_map.shape)
    box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    #print(rpn.anchor_boxes.shape)
    #print(pred_bbox_deltas.shape)
    # anchor_generator = AnchorGenerator()
    # images = ImageList([torch.zeros(3,512,640), torch.zeros(3,512,640)], [(3,512,640),(3,512,640)])
    # anchor_boxes = anchor_generator(images, [feature_map, feature_map])
    # anchor_boxes = anchor_generator([img, img], [feature_map, feature_map])
    # print(type(anchor_boxes), anchor_boxes.shape)
    pred_bbox = box_coder.decode(pred_bbox_deltas.detach(), anchor_boxes)
    print(pred_bbox.shape)
    rpn([feature_map, feature_map])
# anchor_generator = AnchorGenerator()
# images = ImageList(torch.zeros(3,512,640), [(3,512,640)])
# x = anchor_generator(images, [torch.zeros(512,16,20)])
# print(x[0][3,:])
# class RegionProposalNetwork(nn.Sequential):
#     def __init__(self, feature_map_size):
#         in_channels = feature_map_size[2]
#         super().__init__(
#            nn.Conv2d(in_channels, 512, 3),
#            nn.Conv2d(512,)
#            nn.Conv2d(3, 4),
#            nn.ReLU(),
#            nn.Linear(4, 2),
#            nn.ReLU())
