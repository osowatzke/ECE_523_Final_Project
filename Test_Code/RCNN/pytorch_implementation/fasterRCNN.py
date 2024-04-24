from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ClassConstants import ClassConstants
from PathConstants import PathConstants
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from JsonParser import JsonParser
from torch import Tensor
from FlirDataset import FlirDataset
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.sampler import Sampler
import os
import cv2
import math
import random

def collate_fn(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    # print(data)
    images = []
    targets = []
    idx = []
    for sample in data:
        images.append(sample[0])
        targets.append(sample[1])
        idx.append(sample[2])
    return images, targets, idx
    # targets = for sample in data
    # print(data)
    # _, labels, lengths = zip(*data)
    # max_len = max(lengths)
    # n_ftrs = data[0][0].size(1)
    # features = torch.zeros((len(data), max_len, n_ftrs))
    # labels = torch.tensor(labels)
    # lengths = torch.tensor(lengths)

    # for i in range(len(data)):
    #     j, k = data[i][0].size(0), data[i][0].size(1)
    #     features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    # return features.float(), labels.long(), lengths.long()

def save_state(epoch, model, optimizer):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer.state_dict': optimizer.state_dict(),
        'random_state' : random.getstate(),
        'numpy_random_state' : np.random.get_state(),
        'torch_random_state' : torch.random.get_rng_state(),
    })

class CustomSampler(Sampler):
    def __init__(self, indices):
        # super(self).__init__()
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)
    
    # def __len__
        
def runModel():
  
    # Fix random number generator seed for repeatability
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # torch.cuda.set_device(device)

    # Load Faster RCNN model with default weights
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(ClassConstants.LABELS.keys()))
    model.to(device)

    # Load training data
    # json_path = os.path.join(PathConstants.TRAIN_DIR,'index.json')
    # print(json_path)
    # json_parser = JsonParser(json_path)

    # Get boxes and image paths from JSON parser
    # boxes = json_parser.gt_boxes_all
    # labels = json_parser.gt_classes_all
    # img_paths = json_parser.img_paths

    # Number of images to load
    # num_images = 10

    # Create list of targets
    # One entry per image
    # targets = []
    # for i in range(num_images):
    #     d = {}
    #     d['boxes'] = boxes[i].to(device)
    #     d['labels'] = labels[i].to(device)
    #     targets.append(d)

    # targets.to(device)

    # Load training images
    # images = []
    # for i in range(num_images):
    #     img_path = os.path.join(PathConstants.TRAIN_DIR,'data',img_paths[i])
    #     img = plt.imread(img_path)
    #     if (img.ndim == 2):
    #       img = img.reshape((-1,) + img.shape)
    #     else:
    #       img = np.moveaxis(img, 2, 0)
    #     # img.to(device)
    #     # print(img.shape)
    #     #else:
    #     #  img = np.swapaxes(np.swapaxes(img, 0, 1), 1, 2)
    #     #plt.imshow(img)
    #     # plt.show()
    #     images.append(img)
# 
    # images = Tensor(np.array(images))
    # images = images.to(device)
    # images.cuda()
    # print(images.shape)
    #output = model(images, targets)
    #print(output)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0)

    curr_file_path = os.path.dirname(__file__)
    run_dir = os.path.join(curr_file_path,'run')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    model_path = os.path.join(run_dir,'trained_model.pth')

    DO_TRAINING = False
    NUM_EPOCHS = 10
    BATCH_SIZE = 2
    MAX_BATCHES = 250

    train_data = FlirDataset(PathConstants.TRAIN_DIR, device=device)
    data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    # for idx, (img, targets) in enumerate(data_loader):
    #     for box in targets[0]['boxes']:
    #         w = box[2] - box[0] #targets[0]['boxes'] - targets[0]['boxes']
    #         h = box[3] - box[1] #targets[0]['boxes'] - targets[0]['boxes']
    #         if w < 0.01 or h < 0.01:
    #             print(idx, box, w, h)
    #             break


    num_samples = len(train_data)
    num_batches = math.ceil(num_samples/BATCH_SIZE)
    batch_count = 0
    kill_run = False
    #print(num_samples)
    #print(num_batches)

    if DO_TRAINING:
        model.train()
        for epoch in range(NUM_EPOCHS):
            total_loss = 0.0
            print(f'Beginning epoch {epoch+1}/{NUM_EPOCHS}:')
            for idx, (img, targets) in enumerate(data_loader):
                #img = [i for i in img]
                #targets = [{'boxes': boxes, 'labels': labels} for boxes, labels in zip(targets['boxes'], targets['labels'])]
                # print(targets[0])
                optimizer.zero_grad()
                loss_dict = model(img, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                print(f'Batch Loss ({idx + 1}/{num_batches}): {losses.item()}')
                total_loss += losses.item() * len(img)
                batch_count += 1
                if (MAX_BATCHES > 0):
                    if batch_count == MAX_BATCHES:
                        kill_run = True
                        break
            print(f'Total loss: {total_loss/num_samples}')
            if kill_run:
                break
        # for epoch in range(NUM_EPOCHS):
        #     print(f'Beginning epoch: {epoch+1}')
        #     total_loss = 0.0
        #     indexes = torch.randperm(images.shape[0])
        #     images = images[indexes]
        #     targets = [targets[index] for index in indexes]
        #     for idx in range(0, images.shape[0], BATCH_SIZE):
        #         
        #         model.train()
        #         optimizer.zero_grad()
        #         #print(images.dtype)
        #         loss_dist = model(images,targets)
        #         losses = sum(loss for loss in loss_dist.values())
        #         losses.backward()
        #         optimizer.step()
        #         total_loss += losses.item()
        #     print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], loss:{losses.item():.4f}')
        torch.save(model, model_path)
    else:
        model = torch.load(model_path)

    #print(model)
    model.eval()
    image_idx = 5
    pred = model([train_data[0][0]])
    print(ClassConstants.LABELS.keys())
    # pred = [list(i.cpu()) for i in pred]
    targets = [train_data[0][1]]
    pred_class = [list(ClassConstants.LABELS.keys())[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[np.int64((i[0], i[1])), np.int64((i[2]-i[0]+1, i[3]-i[1]+1))] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    true_boxes = [[np.int64((i[0], i[1])), np.int64((i[2]-i[0]+1, i[3]-i[1]+1))] for i in list(targets[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>0.7][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    img = train_data[0][0].cpu().numpy()
    # img = np.uint8(np.moveaxis(img,0,2))
    img = img[0,:,:]
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    for i in range(len(pred_boxes)):
        ax = plt.gca()
        # rect = Rectangle((10, 10), 64, 64, linewidth=1, edgecolor='r', facecolor='none') #np.int64(pred_boxes[i][0]), np.int64(pred_boxes[i][1]), (0,255,0), 1)
        rect = Rectangle(pred_boxes[i][0], pred_boxes[i][1][0], pred_boxes[i][1][1], linewidth=1, edgecolor='r', facecolor='none')
        # rect = Rectangle(np.int64(targets[0][i,0:1]), np.int64(targets[0][i,2]), np.int64(targets[0][i,3]), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        # cv2.putText(img, pred_class[i], np.int64(pred_boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 12, (0,255,0), 1)
        # plt.figure() 
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    # runModel()

    # Fix random number generator seed for repeatability
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    #train_data = FlirDataset(PathConstants.TRAIN_DIR) #, device=device)
    #random_sampler = RandomSampler(data_source=train_data,num_samples=len(train_data))
    #rng_state = torch.random.get_rng_state()
    #idx = list(iter(random_sampler))
    #torch.random.set_rng_state(rng_state)
    #resume_epoch_sampler = CustomSampler(indices=idx)
    sampler = CustomSampler(None)
    train_data = FlirDataset(PathConstants.TRAIN_DIR) #, device=device)
    data_loader = DataLoader(train_data, batch_size=16, collate_fn=collate_fn, shuffle=False, sampler=sampler)
    # idx = list(iter(resume_epoch_sampler))
    for epoch in range(10):
        # sampler.indices = np.arange(len(train_data)) 
        sampler.indices = np.random.permutation(len(train_data))
        for img, target, idx in data_loader:
            print(idx)
            break
            #plt.imshow(img[0,:,:], cmap=plt.get_cmap('gray'))