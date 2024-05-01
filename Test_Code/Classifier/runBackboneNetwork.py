from BackboneNetwork import BackboneNetwork
from FlirDataset import FlirDataset
from PathConstants import PathConstants
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import loadTrainingData as LDT #import CustomSampler, collate_fn
from torch.utils.data.sampler import Sampler
import numpy as np
import torch
import json


def display_results(results):

    nrows, ncols = (1, len(results))
    fig, axs = plt.subplots(nrows, ncols)

    filters_data = [filters[0, :, :].detach().numpy() for filters in results]

    for i in range(len(results)):
        im = axs[i].imshow(filters_data[i])

    plt.show()


def save_results(results, targets):
    
    # Save images
    path = "C:\Users\nicky\OneDrive\Documents\GitHub\ECE_523_Final_Project\FLIR_ADAS_v2\backbone_rgb_train\" + "
    torch.save(results, r'C:\Users\nicky\OneDrive\Documents\GitHub\ECE_523_Final_Project\FLIR_ADAS_v2\backbone_rgb_train\backboneOutput.pt')

    # Save annotations
    # for target in targets:
    #     label = target["labels"]
    #     bbox = target["boxes"]


    # with open('index.json', 'w', encoding='utf-8') as f:
    #     json.dump(targets, f, ensure_ascii=False, indent=4)



def load_data_and_run_backbone(num_images_in):

    # Load data
    dataset = FlirDataset(r'C:\Users\nicky\OneDrive\Documents\GitHub\ECE_523_Final_Project\FLIR_ADAS_v2\images_thermal_train', num_images=num_images_in, downsample=1, device=None)
    sampler = LDT.CustomSampler()
    dataloader = DataLoader(dataset, batch_size=num_images_in, collate_fn=LDT.collate_fn, shuffle=0, sampler=sampler)
    sampler_indices = np.random.permutation(num_images_in)
    sampler.indices = sampler_indices#[num_images_in::]

    # Create backbone
    network = BackboneNetwork()

    # Run data through backbone
    results = []
    targets = []
    for img, targets_ in dataloader:

        img = torch.stack((*img,))
        res = network(img)
        results.extend(res)
        targets.extend(targets_)
        save_results(res)

    # Display results (DEBUG)
    # display_results(results)

    # Save results
    save_results(results, targets)




if __name__ == "__main__":

    load_data_and_run_backbone(100)
   
