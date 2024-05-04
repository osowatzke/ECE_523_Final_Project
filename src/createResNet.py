# Authors: Nick Blanchard and Owen Sowatzke


## DEPENDENCIES ################################################################################
import torch
from torchvision.models import resnet50, ResNet50_Weights
from DataManager import DataManager
from FlirDataset import FlirDataset
from PathConstants import PathConstants
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


## GLOBAL VARIABLES ############################################################################


## HELPER FUNCTIONS ############################################################################
def setup_model():

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    req_layers = list(model.children())[:8]
    resnet = torch.nn.Sequential(*req_layers)

    for param in resnet.named_parameters():
        param[1].requires_grad = True

    return resnet

def run_backbone(model, img_data):

    return model(img_data)


## MAIN FUNCTION ###############################################################################
if __name__ == "__main__":

    # Setup model
    resnet = setup_model()

    # Download data directory
    data_manager = DataManager('train')
    data_manager.download_datasets()
    data_dir = data_manager.get_download_dir()
    PathConstants(data_dir)

    # Run a test image through the resnet backbone
    dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images=1)
    dataloader = DataLoader(dataset)
    for img_batch, gt_bboxes_batch, gt_classes_batch in dataloader:
        img_data_all = img_batch
        gt_bboxes_all = gt_bboxes_batch
        gt_classes_all = gt_classes_batch
        break
    img_data = img_data_all[0]
    img_data = img_data.reshape((1,) + img_data.shape)
    result = resnet(img_data)

    # Display output of resnet backbone
    nrows, ncols = (1, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
    filters_data = [filters.detach().numpy() for filters in result[0]]
    for idx, ax in enumerate(axes):
        ax.imshow(filters_data[idx])
    plt.show()