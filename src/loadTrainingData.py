from FlirDataset import FlirDataset
from PathConstants import PathConstants
from torch.utils.data import DataLoader


def loadTrainingData(num_images, random):
    #####################################
    # num_images = int, number of images to load
    # random     = boolean, True = randomly sample from set
    #####################################

    # Load in test data
    dataset = FlirDataset(PathConstants.TRAIN_DIR)
    dataloader = DataLoader(dataset, batch_size=num_images, shuffle=random)
    
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