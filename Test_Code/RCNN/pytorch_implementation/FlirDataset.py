from JsonParser import JsonParser
from torch.utils.data import Dataset
from PathConstants import PathConstants
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
import cv2

class FlirDataset(Dataset):
    def __init__(self, dir, downsample=1, num_images=-1, device=None):
        file_name = os.path.join(dir,'index.json')
        self.json_parser = JsonParser(file_name)
        self.data_dir = os.path.join(dir,'data')
        self.device = device
        self.images = []
        if num_images == -1:
            num_images = len(self.json_parser.img_paths)
        print('Loading Images. Please be Patient...')
        for img_idx, img_path in enumerate(self.json_parser.img_paths[:num_images]):
            if img_idx % 100 == 0:
                print('%.2f%% Complete' % (100 * img_idx/num_images))
            img = cv2.imread(os.path.join(self.data_dir, img_path))
            if downsample != 1:
                (height, width, _) = img.shape
                dim = (width//downsample, height//downsample)
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                boxes = self.json_parser.gt_boxes_all[img_idx]
                for box_idx in range(len(boxes)):
                    xmin = boxes[box_idx][0]//downsample
                    ymin = boxes[box_idx][1]//downsample
                    xmax = (boxes[box_idx][2] + downsample - 1)//downsample
                    ymax = (boxes[box_idx][3] + downsample - 1)//downsample
                    boxes[box_idx] = torch.Tensor([xmin, ymin, xmax, ymax])
                self.json_parser.gt_boxes_all[img_idx] = boxes
            self.images.append(img)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.json_parser.img_paths[idx]
        img_path = os.path.join(self.data_dir, img_path)
        gt_boxes = self.json_parser.gt_boxes_all[idx]
        gt_classes = self.json_parser.gt_classes_all[idx]
        img = self.images[idx]
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = img.to(dtype=torch.float32)
        if self.device is not None:
            gt_boxes = gt_boxes.to(self.device)
            gt_classes = gt_classes.to(self.device)
            img = img.to(self.device)
        targets = {'boxes': gt_boxes, 'labels': gt_classes}
        return img, targets, idx
    
if __name__ == "__main__":
    PathConstants()
    dataset = FlirDataset(PathConstants.TRAIN_DIR, downsample=4, num_images=1)
    img = dataset[0][0]
    targets = dataset[0][1]
    boxes = targets['boxes']
    labels = targets['labels']
    img_data_all = np.uint8(img.permute(1, 2, 0).numpy())
    plt.imshow(img_data_all)
    print(boxes.shape)
    for gt_box in boxes:
        x = gt_box[0]
        y = gt_box[1]
        w = gt_box[2] - x
        h = gt_box[3] - y
        rect = patches.Rectangle((x, y), w, h, color='red', linewidth=3, fill=False)
        plt.gca().add_patch(rect)
    plt.show()