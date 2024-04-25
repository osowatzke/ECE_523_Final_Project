from JsonParser import JsonParser
from torch.utils.data import DataLoader, Dataset
from PathConstants import PathConstants
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import os
import cv2

class FlirDataset(Dataset):
    def __init__(self, dir, device=None):
        file_name = os.path.join(dir,'index.json')
        self.json_parser = JsonParser(file_name)
        self.data_dir = os.path.join(dir,'data')
        self.device = device
        self.images = []
        for img_path in self.json_parser.img_paths:
            img = cv2.imread(os.path.join(dir, img_path))
            img = torch.from_numpy(img).permute(2, 0, 1)
            img = img.to(dtype=torch.float32)
            self.images.append(self.images)

    def __len__(self):
        return len(self.json_parser.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.json_parser.img_paths[idx]
        img_path = os.path.join(self.data_dir, img_path)
        gt_boxes = self.json_parser.gt_boxes_all[idx]
        gt_classes = self.json_parser.gt_classes_all[idx]
        img = self.images[idx]
        if self.device is not None:
            gt_boxes = gt_boxes.to(self.device)
            gt_classes = gt_classes.to(self.device)
            img = img.to(self.device)
        targets = {'boxes': gt_boxes, 'labels': gt_classes}
        return img, targets, idx
    
if __name__ == "__main__":
    dataset = FlirDataset(PathConstants.TRAIN_DIR)
    dataloader = DataLoader(dataset, batch_size=64)
    for img, target in dataloader:
        print(target)
        break
    for img_batch, gt_bboxes_batch, gt_classes_batch in dataloader:
        img_data_all = img_batch
        gt_bboxes_all = gt_bboxes_batch
        gt_classes_all = gt_classes_batch
        break
    img_data_all = img_data_all[0]
    gt_bboxes_all = gt_bboxes_all[0]
    gt_classes_all = gt_classes_all[0]
    img_data_all = np.uint8(img_data_all.permute(1, 2, 0).numpy())
    plt.imshow(img_data_all)
    for gt_box in gt_bboxes_all:
        x = gt_box[0]
        y = gt_box[1]
        w = gt_box[2] - x
        h = gt_box[3] - y
        rect = patches.Rectangle((x, y), w, h, color='red', linewidth=3, fill=False)
        plt.gca().add_patch(rect)
    plt.show()