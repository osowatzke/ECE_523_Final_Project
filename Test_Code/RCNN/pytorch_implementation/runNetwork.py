import torch
import os
import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from BackboneNetwork        import BackboneNetwork
from CustomDataset          import CustomDataset
from DataManager            import DataManager
from FlirDataset            import FlirDataset
from NetworkTrainer         import NetworkTrainer
from PathConstants          import PathConstants
from RegionProposalNetwork  import RegionProposalNetwork

def collate_fn(data):
    images = []
    targets = []
    for sample in data:
        images.append(sample[0])
        targets.append(sample[1])
    images = torch.cat(images)
    return images, targets

run_backbone = True

# Create Data Manager Object
# Will download dataset if not already available
data_manager = DataManager()
data_manager.download_datasets()

# Create Path Constants Singleton
data_dir = data_manager.get_download_dir()
PathConstants(data_dir)

# Create Dataset Object
dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images = 1)
rpn_dataset = CustomDataset()

# Run images through backbone network
if run_backbone:
    targets = []
    feature_maps = []
    backbone_network = BackboneNetwork()
    for idx in range(len(dataset)):
        img = dataset[idx][0]
        tgt = dataset[idx][1]['boxes']
        img = img.reshape((1,) + img.shape)
        feature_map = backbone_network(img)
        rpn_dataset.append((feature_map, tgt))

torch.manual_seed(0)

rpn = RegionProposalNetwork(dataset[0][0].shape, rpn_dataset[0][0].shape)

if False:
    optimizer = torch.optim.SGD(rpn.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-3)

    network_trainer = NetworkTrainer(
        data       = rpn_dataset,
        model      = rpn,
        optimizer  = optimizer,
        num_epochs = 10,
        batch_size = 8,
        collate_fn = collate_fn)

    network_trainer.train()

file_name = os.path.dirname(__file__)
file_name = os.path.join(file_name, 'run', 'run__2024-05-02_23-08-34','cp__epoch_9_batch_0.pth')
state_dict = torch.load(file_name)
rpn.load_state_dict(state_dict['model_state'])
rpn.eval()

feature_map = rpn_dataset[0][0]
boxes = rpn(feature_map)[0]
boxes = boxes.detach().numpy()
img = dataset[0][0]
img = np.uint8(img.permute(1,2,0))
plt.imshow(img)
for idx, box in enumerate(boxes):
    x = np.round(box[0])
    y = np.round(box[1])
    w = np.round(box[2]) - x + 1
    h = np.round(box[3]) - y + 1
    rect = patches.Rectangle((x, y), w, h, color='red', linewidth=3, fill=False)
    plt.gca().add_patch(rect)
plt.show()





