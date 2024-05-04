from DataManager import DataManager
from FlirDataset import FlirDataset
from PathConstants import PathConstants
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler


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


class CustomSampler(Sampler):
    def __init__(self, indices=None):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)


def loadTrainingData(num_images, random):
    #####################################
    # num_images = int, number of images to load
    # random     = boolean, True = randomly sample from set
    #####################################

    # Download data
    data_manager = DataManager('train')
    data_manager.download_datasets()
    data_dir = data_manager.get_download_dir()
    PathConstants(data_dir)

    # Load in test data
    dataset = FlirDataset(PathConstants.TRAIN_DIR)
    dataloader = DataLoader(dataset, shuffle=random)
    
    # Storing results
    img_data_all = []
    bboxes_data_all = []
    label_data_all = []

    for i in range(num_images):
        img, bboxes, labels = next(iter(dataloader))
        img_data_all.append(img)
        bboxes_data_all.append(bboxes)
        label_data_all.append(labels)

    return (img_data_all, bboxes_data_all, label_data_all)