from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
import torchvision.ops as ops
import torch.nn as nn
import numpy as np
import torch
from torchvision import ops
from FlirDataset import FlirDataset
from PathConstants import PathConstants

class AnchorBoxGenerator():
    def __init__(self, image_size, feature_map_size, sizes=(16,32,64,128,256), aspect_ratios=(0.5,1.0,2.0)):
    
        sizes = np.reshape(np.array(sizes),(1,-1))
        aspect_ratios = np.reshape(np.array(aspect_ratios),(-1,1,))

        w2 = np.round(np.divide(sizes,np.sqrt(aspect_ratios))/2)
        h2 = np.round(sizes*np.sqrt(aspect_ratios)/2)

        print(w2)
        print(h2)

        w2 = np.reshape(w2.ravel(),(1,1,-1))
        h2 = np.reshape(h2.ravel(),(1,1,-1))

        w_sf = image_size[-1] / feature_map_size[-1]
        h_sf = image_size[-2] / feature_map_size[-2]
        
        xc = np.arange(feature_map_size[-1])
        yc = np.arange(feature_map_size[-2])

        xc = np.reshape(xc,(1,-1,1))*w_sf
        yc = np.reshape(yc,(-1,1,1))*h_sf

        xc = np.repeat(xc,yc.shape[0],axis=0)
        yc = np.repeat(yc,xc.shape[1],axis=1)

        xmin = xc - w2
        xmax = xc + w2
        ymin = yc - h2
        ymax = yc + h2

        self.anchor_boxes = torch.Tensor(np.stack((xmin.ravel(), ymin.ravel(), xmax.ravel(),ymax.ravel()),axis=1))

    def sample_anchor_boxes(self, gt_boxes, iou_neg_threshold=0.3, iou_pos_threshold=0.7):
        iou = ops.box_iou(self.anchor_boxes, gt_boxes)
        max_idx = torch.max(iou, dim=0)[1]
        neg_mask = torch.all(iou <= iou_neg_threshold, dim=1)
        pos_mask = torch.all(iou >= iou_pos_threshold, dim=1)
        pos_mask[max_idx] = True

if __name__ == "__main__":
    PathConstants()
    dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images=1)
    anchor_generator = AnchorGenerator(sizes=((64,128,256),))
    images = ImageList(torch.zeros(3,512,640), [(3,512,640)])
    x = anchor_generator(images, [torch.zeros(512,16,20)])
    anchor_generator = AnchorBoxGenerator((3,512,640),(512,16,20))
    print(anchor_generator.sample_anchor_boxes(dataset[0][1]['boxes']))
    # y = anchor_generator.anchor_boxes()
    # print(x[0][9,:])
    # print(x[0][10,:])
    # print(x[0][11,:])
    # print(y[9,:])
    # print(y[10,:])
    # print(y[11,:])
    