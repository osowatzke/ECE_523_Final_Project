import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
import torchvision.ops.boxes as box_ops
import torch

from AnchorBoxUtilities import gen_anchor_boxes
import BoundingBoxUtilities as bbox_utils

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

    # Class constructor
    def __init__(
        self, 
        image_size,
        feature_map_size,
        anchor_box_sizes = (32,64,128),
        aspect_ratios = (0.5,1.0,2.0),
        hidden_layer_size = 512,
        min_proposal_score = 0.5,
        iou_threshold = 0.7):

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
        
        # Get dimensions for neural network layers
        in_channels = feature_map_size[0]
        num_anchors = self.anchor_boxes.shape[0] 

        # Create neural network layers
        self.conv1 = nn.Conv2d(in_channels,hidden_layer_size,3,padding=1)
        self.conv2 = nn.Conv2d(hidden_layer_size,num_anchors,1)
        self.conv3 = nn.Conv2d(hidden_layer_size,4*num_anchors,1)

    def forward(self,feature_map):

        # Run neural network layers
        conv_output = F.relu(self.conv1(feature_map))
        cls_pred = F.sigmoid(self.conv2(conv_output))
        bbox_offsets = self.conv3(conv_output)
        bbox_offsets = bbox_offsets.reshape((-1,4))

        # Get predicted bounding boxes from offsets
        bbox_pred = bbox_utils.centroids_to_corners(bbox_offsets, self.anchor_boxes)

        # Select best bounding boxes
        self.get_best_proposals(bbox_pred)

    def get_best_proposals(self, bbox_pred, scores):

        # Clip bounding boxes to image
        bbox_pred = box_ops.clip_boxes_to_image(bbox_pred, self.image_size[-2:])

        # Remove small boxes
        idx = box_ops.remove_small_boxes(bbox_pred, 1e-3)
        scores = scores[idx]
        bbox_pred = bbox_pred[idx]

        # Remove low scoring boxes
        idx = torch.where(scores >= self.min_proposal_score, 1)
        bbox_pred = bbox_pred[idx]

        # Perform non-maximum suppression
        idx = box_ops.nms(bbox_pred, scores, self.iou_threshold)
        bbox_pred = bbox_pred[idx]

        # Select top N proposals
        


anchor_generator = AnchorGenerator()
images = ImageList(torch.zeros(3,512,640), [(3,512,640)])
x = anchor_generator(images, [torch.zeros(512,16,20)])
print(x[0][3,:])
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
