from torchmetrics.functional.detection.iou import intersection_over_union
from torchmetrics.detection import IntersectionOverUnion
from torch.utils.data import DataLoader

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
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Create list of model predictions
    predictions = []
    for images, _ in dataloader:
        predictions.extend(model(images))

    return predictions

def get_targets(dataset):
    targets = []
    for idx in range(len(dataset)):
        targets.append(dataset[idx][1])
    return targets

def get_iou(predictions, targets, min_score=0):
    
    # Filter model predictions
    updated_predictions = []
    for idx in range(len(predictions)):
        boxes = predictions[idx]['boxes']
        labels = predictions[idx]['labels']
        scores = predictions[idx]['scores']
        boxes  = boxes[scores > min_score]
        labels = labels[scores > min_score]
        scores = scores[scores > min_score]
        updated_predictions.append({
            'boxes'  : boxes,
            'labels' : labels,
            'scores' : scores})

    metric = IntersectionOverUnion()

    met = metric(updated_predictions, targets)
    
    return met['iou']

if __name__ == "__main__":
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from ClassConstants import ClassConstants
    from DataManager import DataManager
    from PathConstants import PathConstants
    from FlirDataset import FlirDataset
    import os

    # Determine the device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Create a faster RCNN model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(ClassConstants.LABELS.keys()))
    model.to(device)

    file_path = os.path.dirname(__file__)
    weights_path = os.path.join(file_path,'weights','built_in','cp__epoch_5_batch_0.pth')
    state_dict = torch.load(weights_path,map_location=device)
    model.load_state_dict(state_dict['model_state'])
    model.eval()

    # Create path constants singleton
    data_manager = DataManager()
    data_manager.download_datasets()
    data_dir = data_manager.get_download_dir()
    PathConstants(data_dir)

    # Create dataset object
    train_data = FlirDataset(PathConstants.TRAIN_DIR, downsample=1, num_images=10, device=device)

    # Get model outputs
    predictions = get_model_outputs(model, train_data, collate_fn=collate_fn)
    targets = get_targets(train_data)
    iou = get_iou(predictions, targets, min_score=0.3)
    print(iou)