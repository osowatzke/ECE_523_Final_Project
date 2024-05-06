from torchmetrics.functional.detection.iou import intersection_over_union
from torchmetrics.detection import IntersectionOverUnion
from torchmetrics.detection import MeanAveragePrecision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fasterRCNN import FasterRCNN
from fasterRCNN import rcnn_collate_fn
import time
import argparse
import torch

# Determine the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    images = []
    targets = []
    for sample in data:
        images.append(sample[0])
        targets.append(sample[1])
    return images, targets

def get_model_outputs(model, dataset, batch_size=1, collate_fn=None):

    # Create Data Loader Object
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=rcnn_collate_fn)

    # Create list of model predictions
    predictions = []
    start_time = time.time()
    for images, _ in dataloader:
        predictions.extend(model(images))

    
    print("Eecution time: {}".format(time.time() - start_time))
    return predictions

def get_targets(dataset):
    targets = []
    for idx in range(len(dataset)):
        targets.append(dataset[idx][1])
    return targets

def filter_predictions(predictions, min_score=0, classes=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)):

    # Filter model predictions
    updated_predictions = []

    # Filter out low scores and wrong classes
    for idx in range(len(predictions)):
        boxes = predictions[idx]['boxes']
        labels = predictions[idx]['labels']
        scores = predictions[idx]['scores']
        boxes  = boxes[scores > min_score]
        labels = labels[scores > min_score]
        scores = scores[scores > min_score]
    
        indices = torch.zeros(labels.size()).to(device)
        for i in range(len(classes)):
            indices = torch.logical_or(indices, labels == classes[i]).to(device)
        boxes  = boxes[indices]
        labels = labels[indices]
        scores = scores[indices]

        updated_predictions.append({
            'boxes'  : boxes,
            'labels' : labels,
            'scores' : scores})

    return updated_predictions

def filter_targets(targets, classes=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)):

    # Filter model predictions
    updated_predictions = []

    # Filter out low scores and wrong classes
    for idx in range(len(predictions)):
        boxes = targets[idx]['boxes']
        labels = targets[idx]['labels']
    
        indices = torch.zeros(labels.size()).to(device)
        for i in range(len(classes)):
            indices = torch.logical_or(indices, labels == classes[i])
        boxes  = boxes[indices]
        labels = labels[indices]

        updated_predictions.append({
            'boxes'  : boxes,
            'labels' : labels})

    return updated_predictions
        
def get_iou(predictions, targets):

    metric = IntersectionOverUnion()

    met = metric(predictions, targets)
    
    return met['iou']

def get_map(predictions, targets, iou_thresholds):

    metric = MeanAveragePrecision(iou_thresholds=iou_thresholds)

    met = metric(predictions, targets)

    print(met)
    return met

if __name__ == "__main__":
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from ClassConstants import ClassConstants
    from DataManager import DataManager
    from PathConstants import PathConstants
    from FlirDataset import FlirDataset
    import os

    # Parse optional input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--class_options', default=1, type=int)
    parser.add_argument('-n', '--num_images', default=50, type=int)
    parser.add_argument('-m', '--map_vals', nargs=3, default=[0.25,0.25,0.25], type=float)
    args = parser.parse_args()

    # Create a Pytorch faster RCNN model
    # model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(ClassConstants.LABELS.keys()))
    # model.to(device)
    file_path = os.path.dirname(__file__)
    # weights_path = os.path.join(file_path,'weights','built_in','cp__epoch_5_batch_0.pth')
    # state_dict = torch.load(weights_path,map_location=device)
    # model.load_state_dict(state_dict['model_state'])
    # model.eval()

    # Create a custom faster RCNN model
    model_ = FasterRCNN(
        image_size       = torch.Size([3, 512, 640]),
        normalize_images = False)
    
    model_.to(device)
    model_.load_state_dict(torch.load(os.path.join(file_path,'weights','custom','cp__epoch_50_batch_0.pth'), map_location=device)['model_state'])
    model_.eval()

    # Create path constants singleton
    data_manager = DataManager()
    data_manager.download_datasets()
    data_dir = data_manager.get_download_dir()
    PathConstants(data_dir)

    if True:

        # Create dataset object
        # train_data = FlirDataset(PathConstants.TRAIN_DIR, downsample=1, num_images=-1, device=device)
        valid_data = FlirDataset(PathConstants.VAL_DIR, downsample=1, num_images=args.num_images, device=device)

        # Get reference bounding boxes
        targets = get_targets(valid_data)

        # Get model outputs
        predictions = get_model_outputs(model_, valid_data, collate_fn=collate_fn)

        # Save predictions and training targets
        torch.save({'predictions': predictions, 'targets': targets}, 'valid_results_custom.pth')
        # imgs = valid_data

    else:
        # Load model data
        model_dict = torch.load('valid_results_custom.pth')

        predictions = model_dict['predictions']
        targets = model_dict['targets']
        imgs = model_dict['imgs']

    map_list = []
    iou_list = []
    for min_score in range(10):

        plotted_image_index = 34

        if args.class_options == 1:
            desired_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        elif args.class_options == 2:
            desired_classes = (3,1)
        elif args.class_options == 3:
            desired_classes = (3,)
        elif args.class_options == 4:
            desired_classes = (1,)

        # desired_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
        filtered_predictions = filter_predictions(predictions, min_score=(0.05*min_score), classes=desired_classes)
        filtered_targets = filter_targets(targets, classes=desired_classes)

        iou = get_iou(filtered_predictions, filtered_targets)

        map = get_map(filtered_predictions, filtered_targets, args.map_vals)
        # map_results.append(map)
        # map = get_map(filtered_predictions, filtered_targets, [0.25, .25, .25])

        # img = model_dict['imgs'][plotted_image_index][0]
        # img_data_all = np.uint8(img.permute(1, 2, 0).numpy())
        # plt.imshow(img_data_all)
        # for i in range((filtered_predictions[plotted_image_index]['boxes'].size()[0])):
        #     x_p = int(filtered_predictions[plotted_image_index]['boxes'][i][0])
        #     y_p = int(filtered_predictions[plotted_image_index]['boxes'][i][1])
        #     w_p = int(filtered_predictions[plotted_image_index]['boxes'][i][2] - x_p)
        #     h_p = int(filtered_predictions[plotted_image_index]['boxes'][i][3] - y_p)
        #     rect = patches.Rectangle((x_p, y_p), w_p, h_p, color='red', linewidth=3, fill=False)
        #     plt.gca().add_patch(rect)
        #     plt.text(x_p,y_p,list(ClassConstants.LABELS.keys())[int(filtered_predictions[plotted_image_index]['labels'][i])],color='white',bbox=dict(facecolor='red', edgecolor='red', boxstyle="Square, pad=0")) # backgroundcolor='red',

        # for i in range((targets[plotted_image_index]['boxes'].size()[0])):
        #     x_p =int(targets[plotted_image_index]['boxes'][i][0])
        #     y_p = int(targets[plotted_image_index]['boxes'][i][1])
        #     w_p = int(targets[plotted_image_index]['boxes'][i][2] - x_p)
        #     h_p = int(targets[plotted_image_index]['boxes'][i][3] - y_p)
        #     rect = patches.Rectangle((x_p, y_p), w_p, h_p, color='blue', linewidth=3, fill=False)
        #     plt.gca().add_patch(rect)
        #     plt.text(x_p,y_p,list(ClassConstants.LABELS.keys())[int(targets[plotted_image_index]['labels'][i])],color='white',bbox=dict(facecolor='blue', edgecolor='blue', boxstyle="Square, pad=0")) # backgroundcolor='red',
        # plt.show()

        map_list.append(map['map_medium'])
        iou_list.append(iou)

        print("IoU and mAP at {}: {}".format(0.05*min_score, iou))

    map = np.array(map_list)
    iou = np.array(iou_list)

    ax = plt.subplot(1, 2, 1)
    ax.plot(0.05*np.arange(10), map)
    ax.set_xlabel('score_threshold')
    ax.set_ylabel('mAP')

    ax = plt.subplot(1, 2, 2)
    ax.plot(0.05*np.arange(10), iou)
    ax.set_xlabel('score_threshold')
    ax.set_ylabel('IOU')

    plt.show()