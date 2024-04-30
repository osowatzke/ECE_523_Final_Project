from FlirDataset import FlirDataset
from PathConstants import PathConstants
from torch.utils.data import DataLoader
import numpy as np


def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    # print(data)
    images = []
    targets = []
    for sample in data:
        images.append(sample[0])
        targets.append(sample[1])
    return images, targets


def loadTrainingData(num_images, random):
    #####################################
    # num_images = int, number of images to load
    # random     = boolean, True = randomly sample from set
    #####################################

    # Load in test data
    dataset = FlirDataset(r'C:\Users\nicky\OneDrive\Documents\GitHub\ECE_523_Final_Project\FLIR_ADAS_v2\images_thermal_train', downsample=1, num_images=10, device=None)
    # dataloader = DataLoader(dataset, batch_size=num_images, collate_fn=collate_fn, shuffle=random)
    
    # # Storing results
    # img_data_all = []
    # bboxes_data_all = []
    # label_data_all = []

    # img_data_all = np.uint8(img_data_all.permute(1, 2, 0).numpy())

    # for i in range(num_images):
    #     img, bboxes = next(iter(dataloader))
    #     img_data_all.append(img)
    #     # bboxes_data_all.append(bboxes)
    #     # label_data_all.append(labels)

    img_data = dataset[0][0]
    targets = dataset[0][1]
    img_data_all = np.uint8(img_data.permute(1, 2, 0).numpy())

    return (img_data_all, targets['boxes'], targets['labels'])