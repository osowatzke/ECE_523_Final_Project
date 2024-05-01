import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ClassConstants import ClassConstants
from PathConstants import PathConstants
from FlirDataset import FlirDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import matplotlib.patches.BoxStyle as BoxStyle
import os

# Determine the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# Create a faster RCNN model
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(ClassConstants.LABELS.keys()))
model.to(device)

# Create dataset object
PathConstants()
train_data = FlirDataset(PathConstants.TRAIN_DIR, downsample=1, num_images=100, device=device)

# Load pretrained weights
file_path = os.path.dirname(__file__)
weights_path = os.path.join(file_path,'weights','cp__epoch_5_batch_0.pth')
state_dict = torch.load(weights_path,map_location=device)
model.load_state_dict(state_dict['model_state'])

# Run model
model.eval()
#print(type(train_data[0][0]))
#test_data = train_data[0][0].reshape((1,) + train_data[0][0].shape)
#print(test_data.shape)
img = train_data[1][0]
pred = model([img])
img = img[0,:,:].detach().numpy()
#print(test_data[0,:,:].shape)
#print(pred)
boxes = pred[0]['boxes'].detach().numpy()
labels = pred[0]['labels'].detach().numpy()
scores = pred[0]['scores'].detach().numpy()
boxes = boxes[scores > 0.5]
labels = labels[scores > 0.5]
boxes = boxes[labels != 0]
labels = labels[labels != 0]
#print(boxes)
#print(ClassConstants.LABELS.keys())
plt.imshow(img)
# plt.show()
for idx, box in enumerate(boxes):
    x = box[0]
    y = box[1]
    w = box[2] - x + 1
    h = box[3] - y + 1
    rect = patches.Rectangle((x, y), w, h, color='red', linewidth=3, fill=False)
    plt.gca().add_patch(rect)
    plt.text(x,y,list(ClassConstants.LABELS.keys())[labels[idx]],color='white',bbox=dict(facecolor='red', edgecolor='red', boxstyle="Square, pad=0")) # backgroundcolor='red',
plt.show()

#print(pred)