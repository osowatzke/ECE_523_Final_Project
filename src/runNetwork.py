import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ClassConstants import ClassConstants
from DataManager import DataManager
from PathConstants import PathConstants
from FlirDataset import FlirDataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fasterRCNN import FasterRCNN
from scipy.io import savemat, loadmat
#import matplotlib.patches.BoxStyle as BoxStyle
import os

# Determine the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# Create Data Manager Object
# Will download dataset if not already available
data_manager = DataManager()
data_manager.download_datasets()

# Create Path Constants Singleton
data_dir = data_manager.get_download_dir()
PathConstants(data_dir)

# Create dataset object
train_data = FlirDataset(PathConstants.TRAIN_DIR, compute_mean_std=False, num_images=200, downsample=1, device=device) # num_images=100, 
# savemat('train_data_statistics.mat',{'image_std' : train_data.std, 'image_mean' : train_data.mean})

mat_dict = loadmat('train_data_statistics.mat')
image_mean = mat_dict['image_mean']
image_std = mat_dict['image_std']

image_mean = torch.tensor(image_mean, dtype=torch.float32)
image_std = torch.tensor(image_std, dtype=torch.float32)
image_mean = image_mean.ravel()
image_std = image_std.ravel()

# Create a faster RCNN model
# model = FasterRCNN(train_data[0][0].shape)
image_size = train_data[0][0].shape
model = FasterRCNN(
    image_size       = image_size,
    normalize_images = False,
    image_mean       = image_mean,
    image_std        = image_std)
model.to(device)

# Load pretrained weights
file_path = os.path.dirname(__file__)
weights_path = os.path.join(file_path,'weights','baseline','cp__epoch_50_batch_0.pth')
state_dict = torch.load(weights_path,map_location=device)
model.load_state_dict(state_dict['model_state'])


# Run model
model.eval()
#print(type(train_data[0][0]))
#test_data = train_data[0][0].reshape((1,) + train_data[0][0].shape)
#print(test_data.shape)
img = train_data[0][0]
pred = model(img.reshape((1,) + img.shape),[train_data[0][1]])[0]
img = img[0,:,:].detach().numpy()
#print(test_data[0,:,:].shape)
#print(pred)
boxes = pred[0]['boxes'].detach().numpy()
labels = pred[0]['labels'].detach().numpy()
scores = pred[0]['scores'].detach().numpy()
boxes = boxes[scores > 0.3]
labels = labels[scores > 0.3]
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