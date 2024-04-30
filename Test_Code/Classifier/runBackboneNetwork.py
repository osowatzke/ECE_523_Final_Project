from BackboneNetwork import BackboneNetwork
from FlirDataset import FlirDataset
from PathConstants import PathConstants
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import loadTrainingData as LDT


def run_network(backbone, im_data):

    # Run a test image through the resnet backbone
    results = []

    for im in im_data:
        results.append(backbone(im))

    return results


def display_results(results):

    nrows, ncols = (1, len(results))
    fig, axs = plt.subplots(nrows, ncols)

    filters_data =[filters[0][0, :, :].detach().numpy() for filters in results]

    for i in range(len(results)):
        im = axs[i].imshow(filters_data[i])

    plt.show()


def save_results(results):
    pass


def load_data_and_run_backbone(num_images):

    # Load data
    img_data, bboxes_data, label_data = LDT.loadTrainingData(num_images, True)

    # Create backbone
    network = BackboneNetwork()

    # Run backbone network
    results = run_network(network, img_data)

    # Display results (DEBUG)
    display_results(results)

    return results, bboxes_data, label_data


if __name__ == "__main__":

    load_data_and_run_backbone(10)
   
