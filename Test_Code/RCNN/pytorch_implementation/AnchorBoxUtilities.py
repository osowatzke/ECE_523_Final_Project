import numpy as np
import torch
from torchvision.ops import box_iou

def gen_anchor_boxes(
    image_size,
    feature_map_size,
    sizes = (32,64,128),
    aspect_ratios = (0.5,1.0,2.0)):

    # Convert sizes and aspect ratio into column and row vectors respectively
    sizes = np.reshape(np.array(sizes),(1,-1))
    aspect_ratios = np.reshape(np.array(aspect_ratios),(-1,1))

    # Compute the width/2 and height/2
    w2 = np.round(np.divide(sizes,np.sqrt(aspect_ratios))/2)
    h2 = np.round(sizes*np.sqrt(aspect_ratios)/2)

    # Convert to multidimensional arrays
    w2 = np.reshape(w2.ravel(),(1,1,-1))
    h2 = np.reshape(h2.ravel(),(1,1,-1))

    # Determine scale factors for the width and height
    w_sf = image_size[-1] / feature_map_size[-1]
    h_sf = image_size[-2] / feature_map_size[-2]
        
    # Determine the centers of bounding box along each dimension
    xc = np.arange(feature_map_size[-1])
    yc = np.arange(feature_map_size[-2])

    # Create multidimensional array and map to original image
    xc = np.reshape(xc,(1,-1,1))*w_sf
    yc = np.reshape(yc,(-1,1,1))*h_sf

    # Repeat points along each dimension to get center of each box
    xc = np.repeat(xc,yc.shape[0],axis=0)
    yc = np.repeat(yc,xc.shape[1],axis=1)

    # Determine the four corners of each bounding boxes
    xmin = xc - w2
    xmax = xc + w2
    ymin = yc - h2
    ymax = yc + h2

    # Convert multidimensional arrays to column vectors
    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmax = xmax.ravel()
    ymax = ymax.ravel()

    # Stack vectors to form Nx4 array of anchor boxes
    anchor_boxes = torch.Tensor(np.stack((xmin, ymin, xmax, ymax), axis=1))
    return anchor_boxes

def select_closest_anchors(all_targets, anchor_boxes, max_iou_thresh, min_iou_thresh):
    all_labels = []
    all_ref_boxes = []
    # print(len(all_targets))
    for targets in all_targets:
        if targets.numel() == 0:
            labels = torch.zeros(anchor_boxes.shape[0])
            ref_boxes = torch.zeros(anchor_boxes.shape)
        else:
            iou = box_iou(targets, anchor_boxes) # N x M for N x 4 and M x 4 inputs
            max_val, max_idx = iou.max(dim=0)
            pos_idx = torch.where(max_val > max_iou_thresh)
            neg_idx = torch.where(max_val < min_iou_thresh)
            labels = torch.full((anchor_boxes.shape[0],), -1)
            labels[pos_idx] = 1
            labels[neg_idx] = 0
            ref_boxes = targets[max_idx]
        all_labels.append(labels)
        all_ref_boxes.append(ref_boxes)
    return all_labels, all_ref_boxes
