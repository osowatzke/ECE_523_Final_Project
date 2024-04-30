import json
import os
import torch

from ClassConstants import ClassConstants
from torch.nn.utils.rnn import pad_sequence
from PathConstants import PathConstants

class JsonParser:
    def __init__(self, path):
        self.load_json_file(path)
        self.read_frames()

    def load_json_file(self, path):
        with open(path, "r") as f:
            self.json_dict = json.load(f)

    def read_bounding_box(self, bounding_box):
        h = float(bounding_box['h'])
        w = float(bounding_box['w'])
        xmin = float(bounding_box['x'])
        ymin = float(bounding_box['y'])
        xmax = xmin + w - 1
        ymax = ymin + h - 1
        bbox = torch.Tensor([xmin, ymin, xmax, ymax])
        return bbox
    
    def read_label(self, label):
        if label in ClassConstants.LABELS:
            return ClassConstants.LABELS[label]
        else:
            return -1
        
    def read_annotation(self, annotation):
        bbox = self.read_bounding_box(annotation['boundingBox'])
        for label in annotation['labels']:
            gt_class = self.read_label(label)
            if (gt_class >= 0):
                self.gt_classes.append(gt_class)
                self.gt_boxes.append(bbox.tolist())

    def read_annotations(self, annotations):
        self.gt_classes = []
        self.gt_boxes = []
        for annotation in annotations:
            self.read_annotation(annotation)

    def get_file_name(self, frame):
        frame_index = frame['videoMetadata']['frameIndex']
        video_id = frame['videoMetadata']['videoId']
        frame_id = frame['datasetFrameId']
        file_name = f'video-{video_id}-frame-{frame_index:>0{6}d}-{frame_id}.jpg'
        self.img_paths.append(file_name)
    
    def read_frame(self, frame):
        self.get_file_name(frame)
        annotations = frame['annotations']
        self.read_annotations(annotations)
        self.gt_classes_all.append(torch.Tensor(self.gt_classes))
        if len(self.gt_boxes) > 0:
            self.gt_boxes_all.append(torch.Tensor(self.gt_boxes))
        else:
            self.gt_boxes_all.append(torch.zeros(0,4))
        
    def read_frames(self):
        self.img_paths = []
        self.gt_classes_all = []
        self.gt_boxes_all = []
        for frame in self.json_dict['frames']:
            self.read_frame(frame)
        self.gt_classes_all = pad_sequence(self.gt_classes_all, batch_first=True, padding_value=-1)
        self.gt_boxes_all = pad_sequence(self.gt_boxes_all, batch_first=True, padding_value=-1)

if __name__ == "__main__":
    in_path = os.path.join(PathConstants.SRC_DIR,'..','FLIR_ADAS_v2','images_thermal_train','index.json')
    parser = JsonParser(in_path)