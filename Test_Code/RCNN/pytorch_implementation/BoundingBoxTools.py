import torchvision.ops as ops
import torch

def decode_boxes(anchor_boxes, offsets):
    xmin = anchor_boxes[:,0]
    ymin = anchor_boxes[:,1]
    xmax = anchor_boxes[:,2]
    ymax = anchor_boxes[:,3]
    wa = xmax - xmin
    ha = ymax - ymin
    xa = xmin + 0.5*wa
    ya = ymin + 0.5*ha
    x = offsets[:,0]*wa + xa
    y = offsets[:,1]*ha + ya
    w = wa*torch.exp(offsets[:,2])
    h = ha*torch.exp(offsets[:,3])
    h2 = 0.5 * h
    w2 = 0.5 * w
    pred_xmin = x - w2
    pred_xmax = x + w2
    pred_ymin = y - h2
    pred_ymax = y + h2
    pred_boxes = torch.stack((pred_xmin, pred_ymin, pred_xmax, pred_ymax))
    return pred_boxes

def encode_boxes(anchor_boxes, gt_boxes):
    xmin = gt_boxes[:,0]
    ymin = gt_boxes[:,1]
    xmax = gt_boxes[:,2]
    ymax = gt_boxes[:,3]
    w = xmax - xmin
    h = ymax - ymin
    x = xmin + 0.5*w
    y = ymin + 0.5*h
    xmin = anchor_boxes[:,0]
    ymin = anchor_boxes[:,1]
    xmax = anchor_boxes[:,2]
    ymax = anchor_boxes[:,3]
    wa = xmax - xmin
    ha = ymax - ymin
    xa = xmin + 0.5*w
    ya = ymin + 0.5*h
    tx = (x - xa)/wa
    ty = (y - ya)/ha
    tw = torch.log(w/wa)
    th = torch.log(h/ha)
    offsets = torch.stock((tx,ty,tw,th))
    return offsets

def get_target_anchor_boxes(anchor_boxes, gt_boxes, low_threshold=0.3, high_treshold=0.7):
    iou = ops.box_iou(gt_boxes,anchor_boxes)
    max_val, max_idx = torch.max(iou, dim=0)
    max_idx[max_val < low_threshold] = -1
    max_idx[low_threshold <= max_val < high_treshold] = -2
    matched_gt_boxes = gt_boxes[max_idx.clamp(min=0)]
    labels = max_idx >= 0
    labels[max_idx == -2] = -1
    return labels, matched_gt_boxes

def sample_anchor_boxes(labels, num_samp=128, positive_fraction=0.5):
    pos_idx = torch.where(labels == 1)[0]
    neg_idx = torch.where(labels == 0)[0]
    max_pos = num_samp*positive_fraction
    num_pos = min(pos_idx.numel(), max_pos)
    max_neg = num_samp - num_pos
    num_neg = min(neg_idx.numel(), max_neg)
    pos_idx = torch.randperm(pos_idx.numel())[:num_pos]
    neg_idx = torch.randperm(neg_idx.numel())[:num_neg]
    pos_samp = torch.zeros(labels.shape, dtype=torch.bool)
    neg_samp = torch.zeros(labels.shape, dtype=torch.bool)
    pos_samp[pos_idx] = True
    neg_samp[neg_idx] = True
    return pos_samp, neg_samp