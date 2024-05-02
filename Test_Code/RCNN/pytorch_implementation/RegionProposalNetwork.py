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
import math

from AnchorBoxUtilities import gen_anchor_boxes, select_closest_anchors
import BoundingBoxUtilities as bbox_utils
from BackboneNetwork import BackboneNetwork

from PathConstants import PathConstants
from FlirDataset import FlirDataset

import pprint
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

def sample_pred(all_labels, batch_size, pos_frac):
    pos_samp = []
    neg_samp = []
    for labels in all_labels:
        pos_idx = torch.where(labels == 1)[0]
        neg_idx = torch.where(labels == 0)[0]
        # print(neg_idx[1000:1010])
        max_pos = batch_size*pos_frac
        num_pos = min(pos_idx.numel(), max_pos)
        num_neg = batch_size - num_pos
        rand_idx = torch.randperm(pos_idx.numel())
        pos_idx = pos_idx[rand_idx]
        rand_idx = torch.randperm(neg_idx.numel())
        neg_idx = neg_idx[rand_idx]
        pos_samp.append(pos_idx[:num_pos])
        neg_samp.append(neg_idx[:num_neg])
    return pos_samp, neg_samp


class RegionProposalNetwork2(nn.Module):

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
                    torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                    if layer.bias is not None:
                        torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

        def forward(self, feature_maps):

            # Create empty list for layer outputs
            # cls_pred = [None] * len(feature_maps)
            # bbox_offsets = [None] * len(feature_maps)

            # Loop over feature maps for each batch
            # for i, feature_map in enumerate(feature_maps):
            shared_conv = F.relu(self.conv1(feature_maps))

            # cls_pred[i] = F.sigmoid(self.conv2(shared_conv))
            cls_pred = self.conv2(shared_conv)
            bbox_offsets = self.conv3(shared_conv)

            return [cls_pred], [bbox_offsets]

    # Class constructor
    def __init__(
        self, 
        image_size,
        feature_map_size,
        anchor_box_sizes = (32,64,128),
        aspect_ratios = (0.5,1.0,2.0),
        hidden_layer_channels = 2048,
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
        
        # print(self.anchor_boxes[:9,:])

        # Get dimensions for neural network layers
        in_channels = feature_map_size[-3]
        num_anchors = len(anchor_box_sizes) * len(aspect_ratios)

        # Create convolutional neural network layers
        self.conv_layers = self.ConvLayers(in_channels, hidden_layer_channels, num_anchors)

    def forward(self,feature_maps,targets):

        # Run convolution layers
        cls_pred, bbox_offsets = self.conv_layers(feature_maps)

        # Format output of convolutional layers to form Nx1 and Nx4 matrices
        cls_pred = self.format_cls_pred(cls_pred)
        bbox_offsets = self.format_bbox_offsets(bbox_offsets)

        #print(cls_pred.shape)
        #print(bbox_offsets.shape)

        # Get bounding box from offsets
        bbox_pred = bbox_utils.centroids_to_corners(bbox_offsets, self.anchor_boxes)

        # Reshape to correct dimensions
        cls_pred = cls_pred.reshape(len(feature_maps), -1)
        bbox_pred = bbox_pred.reshape(len(feature_maps), -1, 4)

        #print(cls_pred)
        #print(bbox_pred)

        # Get labels 
        cls_truth, bbox_truth = select_closest_anchors(targets, self.anchor_boxes, 0.5, 0.3)

        # print(bbox_truth[0][cls_truth[0] > 0])
        # print(bbox_truth[1][cls_truth[1] > 0])
        #print(cls_truth[0].shape)
        #print(bbox_truth[0][cls_truth[0] > 0])

        # print(len(cls_truth))
        # print(len(bbox_truth))
        bbox_off_truth = torch.zeros((len(feature_maps),) + bbox_truth[0].shape)
        for i, bbox in enumerate(bbox_truth):
            bbox_off_truth[i,:,:] = bbox_utils.corners_to_centroid(bbox, self.anchor_boxes)

        bbox_offsets = bbox_offsets.reshape(bbox_off_truth.shape)
        # print(bbox_off_truth.shape)
        # print(bbox_offsets.shape)
        # print(cls_truth[0].shape)
        # print(cls_pred[0].shape)

        bbox_loss, cls_loss = self.get_loss(bbox_offsets, bbox_off_truth, cls_pred, cls_truth)

        # Select best bounding boxes
        bbox_pred, cls_pred = self.get_best_proposals(bbox_pred, cls_pred)

        # print(bbox_pred)
        # print(cls_pred)
        
        losses = {
            'bbox' : bbox_loss,
            'cls'  : cls_loss
        }

        return bbox_pred, losses
        # pprint.pp(labels)
        # pprint.pp(bbox_truth[labels == 1])
        #print(labels)
        #print(bbox_truth)

    def format_cls_pred(self, cls_pred):
        cls_pred = torch.stack(cls_pred)
        # print(cls_pred.shape)
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
        
        # print(bbox_pred)

        for boxes, scores in zip(bbox_pred, cls_pred):

            # Get N boxes with highest score before non-max suppression
            _, idx = scores.topk(pre_nms_top_n)
            boxes, scores = boxes[idx], scores[idx]

            # print(boxes)

            scores = torch.sigmoid(scores)

            # Clip bounding boxes to image
            boxes = box_ops.clip_boxes_to_image(boxes, self.image_size[-2:])
            # print(boxes)

            # Remove small boxes
            idx = box_ops.remove_small_boxes(boxes, 1e-3)
            boxes, scores = boxes[idx], scores[idx]

            # print(boxes)

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

    def get_loss(self, bbox_off_pred, bbox_off_truth, cls_pred, cls_truth):
        
        # print(bbox_off_pred.shape)
        # print(bbox_off_truth.shape)
        # print(cls_pred.shape)
        # print(cls_truth.shape)

        pos_samp, neg_samp = sample_pred(cls_truth, 128, 0.5)

        # print(neg_samp[0].sort()[0])

        for idx in range(len(pos_samp)):
            pos_samp[i] = pos_samp[i] + bbox_off_pred.shape[1] * idx
            neg_samp[i] = neg_samp[i] + bbox_off_pred.shape[1] * idx

        pos_samp = torch.cat(pos_samp).sort()[0]
        neg_samp = torch.cat(neg_samp).sort()[0]

        # print(pos_samp.shape)
        # print(pos_samp.sort()[0])
        # print(neg_samp.shape)
        # print(neg_samp.sort()[0])

        # print(pos_samp.shape)
        # print(neg_samp.shape)
        all_samp = torch.cat([pos_samp, neg_samp])

        bbox_off_pred = bbox_off_pred.reshape(-1,4)
        bbox_off_truth = bbox_off_truth.reshape(-1,4)

        # print(bbox_off_pred)
        # print(bbox_off_truth)

        # cls_pred = torch.cat(cls_pred)
        cls_pred = cls_pred.ravel()
        cls_truth = torch.cat(cls_truth)

        bbox_loss = F.smooth_l1_loss(
            bbox_off_pred[pos_samp],
            bbox_off_truth[pos_samp],
            beta=1/9,
            reduction="sum",
        ) / (all_samp.numel())

        # print(cls_pred[0].shape)
        # print(cls_truth[0].shape)
        cls_loss = F.binary_cross_entropy_with_logits(cls_pred[all_samp], cls_truth[all_samp].type(torch.float))

        return bbox_loss, cls_loss
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


def run_network(images, features, targets):

    anchor_box_sizes = ((32,64,128),)
    aspect_ratios = ((0.5, 1.0, 2.0),)
    
    #anchor_box_sizes = (anchor_box_sizes,)
    #aspect_ratios = (aspect_ratios,)*len(anchor_box_sizes)

    anchor_generator = AnchorGenerator(anchor_box_sizes, aspect_ratios)
    #print(len(anchor_generator.cell_anchors))
    #print(anchor_generator.cell_anchors[0].shape)

    torch.manual_seed(0)

    rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=RPNHead(feature_maps[0].shape[1],9),
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.3,
        batch_size_per_image=128,
        positive_fraction=0.5,
        pre_nms_top_n ={'training': 512, 'test': 512},
        post_nms_top_n={'training': 128, 'test': 128},
        nms_thresh=0.7,
        score_thresh=1e-3)
    
    rpn.train()

    torch.manual_seed(0)

    features = list(features.values())
    # print(features[0].shape)

    #head = rpn.head(2048,9)
    objectness, pred_bbox_deltas = rpn.head(features)
    anchors = anchor_generator(images, features)
    # print(anchors[0].shape)
    #A = anchors.reshape(3, -1, 3, 4)
    #A.permute(0,2,1,3)
    # A = A.reshape(-1,4)
    # print(A[:9,:])

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)

    #print(objectness.shape)
    #print(pred_bbox_deltas.shape)
    # 
    # print(objectness.shape)
    # print(pred_bbox_deltas.shape)
    # print(anchors[0].shape)

    box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    proposals = box_coder.decode(pred_bbox_deltas.detach(), anchors)
    
    '''
    rel_codes = pred_bbox_deltas.detach()
    boxes = anchors

    boxes_per_image = [b.size(0) for b in boxes]
    # print(boxes_per_image)
    concat_boxes = torch.cat(boxes, dim=0)
    box_sum = 0
    for val in boxes_per_image:
        box_sum += val
    if box_sum > 0:
        rel_codes = rel_codes.reshape(box_sum, -1)
    # print(concat_boxes.shape)
    # print(rel_codes.shape)
    
    boxes = concat_boxes
    boxes = boxes.to(rel_codes.dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # print(widths.shape)
    # print(widths)
    # print(heights)
    # print(ctr_x)
    # print(ctr_y)

    wx, wy, ww, wh = (1.0,)*4
    dx = rel_codes[:, 0::4] / wx
    dy = rel_codes[:, 1::4] / wy
    dw = rel_codes[:, 2::4] / ww
    dh = rel_codes[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    lim = math.log(1000.0/16)
    dw = torch.clamp(dw, max=lim)
    dh = torch.clamp(dh, max=lim)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    # Distance from center to box's corner.
    c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

    pred_boxes1 = pred_ctr_x - c_to_c_w
    pred_boxes2 = pred_ctr_y - c_to_c_h
    pred_boxes3 = pred_ctr_x + c_to_c_w
    pred_boxes4 = pred_ctr_y + c_to_c_h
    pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
    #print(pred_boxes)
    '''
    proposals = proposals.view(num_images, -1, 4)
    #print(proposals)
    #print(objectness)
    boxes, scores = rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
    # print(boxes)
    '''
    num_images = proposals.shape[0]
    device = proposals.device
    # do not backprop through objectness
    objectness = objectness.detach()
    objectness = objectness.reshape(num_images, -1)

    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    levels = levels.reshape(1, -1).expand_as(objectness)

    # select top_n boxes independently per level before applying nms
    top_n_idx = rpn._get_top_n_idx(objectness, num_anchors_per_level)

    image_range = torch.arange(num_images, device=device)
    batch_idx = image_range[:, None]

    # print(proposals)

    objectness = objectness[batch_idx, top_n_idx]
    levels = levels[batch_idx, top_n_idx]
    proposals = proposals[batch_idx, top_n_idx]

    objectness_prob = torch.sigmoid(objectness)

    final_boxes = []
    final_scores = []
    for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, images.image_sizes):

        # print(boxes)

        boxes = box_ops.clip_boxes_to_image(boxes, img_shape)

        # print(boxes)

        # remove small boxes
        keep = box_ops.remove_small_boxes(boxes, rpn.min_size)
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # print(boxes)

        # remove low scoring boxes
        # use >= for Backwards compatibility
        keep = torch.where(scores >= rpn.score_thresh)[0]
        boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

        # non-maximum suppression, independently done per level
        keep = box_ops.batched_nms(boxes, scores, lvl, rpn.nms_thresh)

        # keep only topk scoring predictions
        keep = keep[: rpn.post_nms_top_n()]
        boxes, scores = boxes[keep], scores[keep]

        final_boxes.append(boxes)
        final_scores.append(scores)

    #print(final_boxes)
    #print(final_scores)
    '''

    #print(boxes)
    #print(scores)

    losses = {}
    if targets is None:
        raise ValueError("targets should not be None")
    labels, matched_gt_boxes = rpn.assign_targets_to_anchors(anchors, targets)

    '''
    labels = []
    matched_gt_boxes = []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        gt_boxes = targets_per_image["boxes"]

        if gt_boxes.numel() == 0:
            # Background image (negative example)
            device = anchors_per_image.device
            matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
            labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
        else:
            match_quality_matrix = rpn.box_similarity(gt_boxes, anchors_per_image)
            # print(match_quality_matrix)
            matched_idxs = rpn.proposal_matcher(match_quality_matrix)

            # print(torch.where(matched_idxs >= 0))
            # print(matched_idxs[matched_idxs >= 0])
            # print(matched_idxs[1365])
            # print(matched_idxs[1413])

            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == rpn.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0.0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == rpn.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1.0

        labels.append(labels_per_image)
        matched_gt_boxes.append(matched_gt_boxes_per_image)
    '''
    # print(matched_gt_boxes[0][labels[0] > 0])
    # print(matched_gt_boxes[0][labels[0] > 0])
    # print(matched_gt_boxes[1][labels[1] > 0])

    regression_targets = box_coder.encode(matched_gt_boxes, anchors)

    # print(regression_targets)

    '''
    loss_objectness, loss_rpn_box_reg = rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    '''

    # print(torch.where(labels[0] == 0)[0].shape)
    # print(torch.where(labels[1] == 0)[0].shape)
    # print(torch.where(labels[0] == 0)[0][1000:1010])
    # print(torch.where(labels[1] == 0)[0][1000:1010])
    sampled_pos_inds, sampled_neg_inds = rpn.fg_bg_sampler(labels)
    sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
    sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
    # print(sampled_pos_inds.shape)
    # print(sampled_pos_inds)
    # print(sampled_neg_inds.shape)
    # print(sampled_neg_inds)

    sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

    objectness = objectness.flatten()

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    # print(pred_bbox_deltas)
    # print(regression_targets)

    loss_rpn_box_reg = F.smooth_l1_loss(
        pred_bbox_deltas[sampled_pos_inds],
        regression_targets[sampled_pos_inds],
        beta=1 / 9,
        reduction="sum",
    ) / (sampled_inds.numel())

    loss_objectness = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])


    losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }
    return boxes, losses

def fg_bg_sampler(matched_idxs, batch_size_per_image=128, positive_fraction=0.5):
    pos_idx = []
    neg_idx = []
    for matched_idxs_per_image in matched_idxs:
        positive = torch.where(matched_idxs_per_image >= 1)[0]
        negative = torch.where(matched_idxs_per_image == 0)[0]

        num_pos = int(batch_size_per_image * positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = batch_size_per_image - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        # randomly select positive and negative examples
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx_per_image = positive[perm1]
        neg_idx_per_image = negative[perm2]

        # create binary mask from indices
        pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
        neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

        pos_idx_per_image_mask[pos_idx_per_image] = 1
        neg_idx_per_image_mask[neg_idx_per_image] = 1

        pos_idx.append(pos_idx_per_image_mask)
        neg_idx.append(neg_idx_per_image_mask)

    return pos_idx, neg_idx

if __name__ == "__main__":
    # anchor_generator = AnchorGenerator()
    # images = ImageList(torch.zeros(2,3,512,640), [(3,512,640),(3,512,640)])
    # anchor_boxes = anchor_generator(images, [torch.zeros(2,512,16,20)])
    # #print(len(anchor_boxes))
    # 
    # PathConstants()
    # dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images=1)
    # backbone = BackboneNetwork()
    # img = dataset[0][0].view((1,) + dataset[0][0].shape)
    # print(dataset[0][1])
    # targets = dataset[0][1]['boxes']
    # targets = [targets, targets]
    # print(targets)
    # feature_map = backbone(img)
    # # region_prosal_network = RegionProposalNetwork(img.shape, feature_map.shape)
    # # print(region_prosal_network.anchor_boxes.shape)
    # # region_prosal_network(feature_map)
    # #region_proposal_network = RegionProposalNetwork
# 
    # head = RPNHead(feature_map.shape[1], 9)
    # objectness, pred_bbox_deltas = head.forward([feature_map, feature_map])
    # # print(objectness[0].shape)
    # # print(pred_bbox_deltas)
    # num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    # num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    # print(num_anchors_per_level_shape_tensors)
    # print(num_anchors_per_level)
    # objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # #print(objectness.shape)
    # #print(pred_bbox_deltas.shape)
    # rpn = RegionProposalNetwork(img.shape,feature_map.shape)
    # box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    # #print(rpn.anchor_boxes.shape)
    # #print(pred_bbox_deltas.shape)
    # # anchor_generator = AnchorGenerator()
    # # images = ImageList([torch.zeros(3,512,640), torch.zeros(3,512,640)], [(3,512,640),(3,512,640)])
    # # anchor_boxes = anchor_generator(images, [feature_map, feature_map])
    # # anchor_boxes = anchor_generator([img, img], [feature_map, feature_map])
    # # print(type(anchor_boxes), anchor_boxes.shape)
    # pred_bbox = box_coder.decode(pred_bbox_deltas.detach(), anchor_boxes)
    # print(pred_bbox.shape)
    # rpn([feature_map], [targets[0]])

    num_images = 2

    PathConstants()
    dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images=num_images)
    # print(len(dataset))
    backbone = BackboneNetwork()

    images = []
    targets = []
    feature_maps = []
    for i in range(num_images):
        img = dataset[i][0]
        img = img.reshape((1,) + img.shape)
        images.append(img)
        targets.append(dataset[i][1]['boxes'])
        feature_map = backbone(img)
        feature_maps.append(feature_map)

    
    torch.manual_seed(0)

    rpn = RegionProposalNetwork2(images[0].shape, feature_maps[0].shape) 

    optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-3)

    torch.manual_seed(0)

    num_epochs = 10
    batch_size = 2
    num_batches = math.ceil(num_images/batch_size)
    for epoch in range(num_epochs):
        idx = torch.randperm(num_images)
        # torch.manual_seed(0)
        for batch in range(num_batches):
            s = batch*batch_size
            e = s + batch_size
            e = max(e, num_images)
            batch = []
            tgts = []
            for i in idx[s:e]:
                batch.append(feature_maps[i])
                tgts.append(targets[i])
            optimizer.zero_grad()
            #_, loss_dict = rpn(batch, tgts)
            _, loss_dict = rpn(torch.concat(batch),tgts)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
        print(epoch, losses.item())
        print(torch.get_rng_state()[:10])

    print()

    # torch.save(rpn,'rpn.pth')
    
    # images = ImageList(torch.Tensor(images))

    anchor_box_sizes = ((32,64,128),)
    aspect_ratios = ((0.5,1.0,2.0),)

    # anchor_box_sizes = (32,64,128),
    # aspect_ratios = (0.5,1.0,2.0),
    # 
    # anchor_box_sizes = (anchor_box_sizes)
    # aspect_ratios = (aspect_ratios,)*len(anchor_box_sizes)

    anchor_generator = AnchorGenerator(anchor_box_sizes, aspect_ratios)

    torch.manual_seed(0)

    rpn = RegionProposalNetwork(
        anchor_generator=anchor_generator,
        head=RPNHead(feature_maps[0].shape[1],9),
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.3,
        batch_size_per_image=128,
        positive_fraction=0.5,
        pre_nms_top_n ={'training': 512, 'test': 512},
        post_nms_top_n={'training': 128, 'test': 128},
        nms_thresh=0.7,
        score_thresh=0.5)
    
    rpn.train()

    optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-3)
    
    torch.manual_seed(0)

    num_epochs = 10
    batch_size = 2
    num_batches = math.ceil(num_images/batch_size)
    # transform = GeneralizedRCNNTransform()
    for epoch in range(num_epochs):
        idx = torch.randperm(num_images)
        # torch.manual_seed(0)
        for batch in range(num_batches):
            s = batch*batch_size
            e = s + batch_size
            e = max(e, num_images)
            imgs = []
            #batch = {}
            tgts = []
            batch = []
            for i in idx[s:e]:
                imgs.append(images[i])
                #batch[i] = feature_maps[i]
                batch.append(feature_maps[i])
                tgts.append({'boxes':targets[i]})
            imgs = torch.cat(imgs)
            batch = {'0' : torch.cat(batch)}
            num_imgs = e - s
            optimizer.zero_grad()
            # print(targets[i].shape)
            # print(feature_maps[i].shape)
            # args = [ImageList(images[i],(images[i].shape[-2:],)),{'0':feature_maps[i]},[{'boxes':targets[i]}]]
            args = [ImageList(imgs,(images[i].shape[-2:],)*num_imgs),batch,tgts]
            #_, loss_dict = run_network(*args)
            _, loss_dict = rpn(*args)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
        print(epoch, losses.item())
        print(torch.get_rng_state()[:10])

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
