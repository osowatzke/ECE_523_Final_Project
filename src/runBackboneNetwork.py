from BackboneNetwork import BackboneNetwork
from FlirDataset import FlirDataset
from PathConstants import PathConstants
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def run_network(backbone, im_data):

    # Run a test image through the resnet backbone
    results = []

    for im in im_data:
        results.append(backbone(im))

    return results


def load_data(num_images):

    # Load in test data
    dataset = FlirDataset(PathConstants.TRAIN_DIR)
    dataloader = DataLoader(dataset, batch_size=num_images)
    
    # Storing results
    img_data_all = []
    label_data_all = []

    for i in range(num_images):
        img, bboxes, labels = next(iter(dataloader))
        img_data_all.append(img)
        label_data_all.append(labels)

    return img_data_all


def display_results(results):

    nrows, ncols = (1, len(results))
    fig, axs = plt.subplots(nrows, ncols)

    filters_data =[filters[0].detach().numpy() for filters in results]

    for i in range(len(results)):
        im = axs.imshow(filters_data[i])

    plt.show()


if __name__ == "__main__":

    # Number of images
    num_images = 1

    # Load data
    img_data = load_data(num_images)

    # Create backbone
    network = BackboneNetwork()

    # Run backbone network
    results = run_network(network, img_data)

    # Display results
    display_results(results)
