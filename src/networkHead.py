import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from FlirDataset import FlirDataset
from PathConstants import PathConstants
import loadTrainingData as LTD
from torch.utils.data.sampler import Sampler
from BackboneNetwork import BackboneNetwork
import numpy as np
import math
from torchvision.models.detection.roi_heads import fastrcnn_loss
from BoundingBoxUtilities import corners_to_centroid, centroids_to_corners
from ClassConstants import ClassConstants

from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign

from torch.utils.data      import DataLoader
from RoiHeadsDataset       import RoiHeadsDataset
from RegionProposalNetwork import *

device = torch.device('cpu')
# torch.cuda.set_device(0)

IMG_HEIGHT = 7
IMG_WIDTH = 7
IMG_CHANNELS = 2048

BATCH_SIZE = 32
NUM_CLASSES = 16

BACKBONE_FLAG = True    # True = run data through backbone before classifier


class headNet(nn.Module):


    def __init__(self, num_images_in):
        super(headNet, self).__init__()

        # Class labels
        classes = ClassConstants()
        self.labels = classes.LABELS

        # First fully connected layer
        self.fc1 = nn.Linear(IMG_HEIGHT*IMG_WIDTH*IMG_CHANNELS, 1024, bias=True)

        # Second fully connected layer
        self.fc2 = nn.Linear(1024, 1024, bias=True)

        # Dropout
        self.drp = nn.Dropout(p=0.5)

        # Batch normalization
        self.m = nn.BatchNorm1d(1024)

        # Third fully connected layer (for classification)
        self.fc3 = nn.Linear(1024, len(self.labels), bias=True)
        # self.fc3 = nn.Linear(1024, 16, bias=True)

        # Fourth fully connected layer (for regression)
        self.fc4 = nn.Linear(1024, 4*len(self.labels), bias=True)
        # self.fc4 = nn.Linear(1024, 4*16, bias=True)

        # Softmax
        self.softmax = torch.nn.Softmax(dim = 1)

        # Initialize weights
        # (already done by Pytorch automatically)

        # Create data loaders for our datasets; shuffle for training, not for validation
        self.numSamples = num_images_in

    
    def initDataLoader(self):
        sampler = LTD.CustomSampler()
        valid_sampler = LTD.CustomSampler()
        sampler.indices = range(int(self.numSamples - self.numSamples % BATCH_SIZE))
        valid_sampler.indices = range(int(2* BATCH_SIZE))
        self.training_set = FlirDataset(r"C:\Users\nicky\OneDrive\Documents\GitHub\ECE_523_Final_Project\FLIR_ADAS_v2\images_thermal_train", num_images=int(self.numSamples - self.numSamples % BATCH_SIZE), downsample=1, device=None)
        self.validation_set = FlirDataset(r"C:\Users\nicky\OneDrive\Documents\GitHub\ECE_523_Final_Project\FLIR_ADAS_v2\images_thermal_val", num_images=int(2*BATCH_SIZE), downsample=1, device=None)
        self.training_loader = torch.utils.data.DataLoader(self.training_set, batch_size=BATCH_SIZE, collate_fn=LTD.collate_fn, shuffle=False, sampler=sampler)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_set, batch_size=BATCH_SIZE, collate_fn=LTD.collate_fn, shuffle=False, sampler=valid_sampler)
        print('Training set has {} instances'.format(len(self.training_set)))
        print('Validation set has {} instances'.format(len(self.validation_set)))


    def forward(self, x, labels, regression_targets):

        # Pass input through layers, with relu and max pooling nonlinearities
        # x = x.double()

        # Shared backbone
        x = self.fc1(x)                     
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        # Normalization and dropout
        x = self.m(x)    # Normalization                

        # Classification
        c = self.fc3(x)                     
        
        # Regression
        r = self.fc4(x)                     

        # Softmax
        # output = F.log_softmax(x, dim=1)
        # output = self.softmax(x)

        # Losses
        labels = labels.type(torch.LongTensor).to(device)
        classification_loss = F.cross_entropy(c, labels)

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        N, num_classes = c.shape
        r = r.reshape(N, r.size(-1) // 4, 4)

        box_loss = F.smooth_l1_loss(
            r[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()
        
        return c, r, classification_loss, box_loss
    

    def lossSumFn(inputs):
        # inputs is the exact output of the forward path
        return sum((inputs[2], inputs[3]))


    def trainOneEpoch(self, epoch_index, tb_writer, backbone=None):

        # Loss metrics
        running_loss = 0.
        last_loss = 0.

        # Specify optimizer
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, eps=1e-08, weight_decay=5e-3)

        # For each batch
        for i, data in enumerate(self.training_loader):

            # Zero gradients before each batch
            optimizer.zero_grad()
           
            # Run network
            outputs, classes_, boxes_ = self.runNetwork(data, backbone)

            # Calculate loss and gradients
            # classes_ = classes_.type(torch.LongTensor).to(device)
            loss_tuple = (outputs[2], outputs[3])
            total_loss = sum(loss for loss in loss_tuple)
            total_loss.backward()

            # Clip gradient
            # torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            
            # Adjust weights
            optimizer.step()

            # Gather data and report
            running_loss += total_loss.item()
            if i > 0 and i % 5 == 0:
                last_loss = running_loss / i
                print(' batch {}  train classloss: {} boxloss: {}'.format(i + 1, loss_tuple[0].item(), loss_tuple[1].item()))
                tb_x = epoch_index * self.numSamples + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0
            
        return running_loss / i


    def runTraining(self, num_epochs=5):

        # Initialize params
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        # writer = SummaryWriter('runs/fashion_trainer_fashion_trainer_20240503_014417')
        epoch_number = 0
        best_vloss = 1000000.0

        # Setup backbone if necessary
        backbone=None
        if BACKBONE_FLAG:
            backbone = BackboneNetwork()

        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Track gradient
            self.train(True)
            avg_loss = self.trainOneEpoch(epoch_number, writer, backbone=backbone)

            running_vloss = 0.0
            self.eval()

            # Disable gradient computation to save mem
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):

                    voutputs, vclasses_, vlabels_, vboxes_ = self.runNetwork(vdata, backbone)

                    # Calculate loss and gradients
                    # vclasses_ = vclasses_.type(torch.LongTensor).to(device)
                    vloss_tuple = (voutputs[2], voutputs[3])
                    vloss = sum(loss for loss in vloss_tuple)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print(' batch {}  valid classloss: {} boxloss: {}'.format(i + 1, vloss_tuple[0].item(), vloss_tuple[1].item()))
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
            print("")

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(self.state_dict(), model_path)

            epoch_number += 1

    
    def runNetwork(self, data, backbone):

        # Load data
        inputs, labels = data

        # Run data through backbone
        if backbone:
            imgs = torch.stack((*inputs,))
            features = (backbone(imgs))

        # Number of ROIs (removing ROIs with class -1)
        numROIs = 0
        for j in range(len(labels)):
            for p in range(len(labels[j]["labels"])):
                numROIs = numROIs + int(labels[j]["labels"][p] != -1.)

        # Hold all data
        labels_ = torch.zeros([numROIs, len(self.labels)]).to(device)
        labels_db = np.empty((len(labels), len(self.labels))) # DEBUG
        labels_db[:,3] = 1
        classes_ = torch.zeros([numROIs]).to(device)
        imgs_ = torch.zeros([numROIs, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH]).to(device)
        boxes_ = torch.zeros([numROIs, 4]).to(device)
        boxes_db = torch.zeros([numROIs, 5]).to(device)

        # ROI Pooling
        # For each image in the batch
        totalROIs = 0
        k = 0
        for label in labels:

            # Number of ROIs in this image
            numROIs_this_img = 0
            for p in range(len(label["labels"])):
                numROIs_this_img = numROIs_this_img + int(label["labels"][p] != -1.)
        
            # For each ROI in image
            j = 0
            numTrueROIs = 0
            temp = torch.zeros(numROIs_this_img, 4).to(device)
            for j in range(len(label["labels"])):

                # Determine class
                class_ = label["labels"][j]

                # If it's not class -1:
                if class_ != torch.tensor(-1.).to(device):

                    # Creating ideal output
                    classes_[totalROIs] = int(class_)

                    # Extracting bounding boxes
                    # boxes_[totalROIs, :] = label["boxes"][j].to(device)
                    boxes_[totalROIs, :] = torch.tensor([0, 0, 1, 1])
                    temp[numTrueROIs,:] = label["boxes"][j]
                    boxes_db[totalROIs] = torch.cat((torch.tensor([k]).to(device), (label["boxes"][j]).to(device)), 0)

                    numTrueROIs = numTrueROIs + 1
                    totalROIs = totalROIs + 1
            k = k + 1
        
        # Experimental: Pytorch RoI Pooling
        imgs_pyt = torchvision.ops.roi_pool(features.to(device), boxes_db.to(device), (7, 7), 0.03125)

        # Re-format feature space to make it compatible with FCL dimensions
        features = torch.flatten(imgs_pyt, start_dim = 1)

        # Run batch through network
        outputs = self(features, classes_, boxes_)

        return outputs, classes_, boxes_

    def testClassifier(self):

        # Disable gradient computation to save mem
        with torch.no_grad():

            # Setup backbone if necessary
            backbone=None
            if BACKBONE_FLAG:
                backbone = BackboneNetwork()

            # For each batch
            for i, tdata in enumerate(self.training_loader):

                toutputs, tclasses_, tboxes_ = self.runNetwork(tdata, backbone)

                # Determine predicted classes
                predOutputs = torch.zeros([toutputs[0].size(dim=0)]).to(device)
                totalAcc = 0
                for r in range(toutputs[0].size(dim=0)):
                    predOutputs[r] = torch.argmax(toutputs[0][r])
                    totalAcc = totalAcc + int(predOutputs[r] == tclasses_[r])
                    # print("Pred: {}, Act: {}".format(int(predOutputs[r]), tclasses_[r]))
                print("Total accuracy is: {}%".format(totalAcc * 100 / toutputs[0].size(dim=0)))

                conf = ignite.metrics.confusion_matrix.ConfusionMatrix(num_classes=17)
                print(conf(predOutputs_, tclasses_))


def create_roi_heads_network(feature_map_size, use_built_in_roi_heads=False):
    
    if not use_built_in_roi_heads:
        print("WARNING: Custom ROI Heads Network has not been fully integrated")
    
    box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    out_channels = feature_map_size[1]
    box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

    num_classes = len(ClassConstants.LABELS.keys())
    representation_size = 1024
    box_predictor = FastRCNNPredictor(representation_size, num_classes)

    bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)

    roi_heads = RoIHeads(
        box_roi_pool         = box_roi_pool,
        box_head             = box_head,
        box_predictor        = box_predictor,
        # Faster R-CNN training
        fg_iou_thresh        = 0.5,
        bg_iou_thresh        = 0.5,
        batch_size_per_image = 512,
        positive_fraction    = 0.25,
        bbox_reg_weights     = bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh         = 0.05,
        nms_thresh           = 0.3,
        detections_per_img   = 100)

    return roi_heads


def create_roi_dataset(rpn, dataset, rpn_dataset, use_built_in_roi_heads=False, device=None):

    if not use_built_in_roi_heads:                 
        print("WARNING: Custom ROI Heads Network has not been fully integrated")

    if len(rpn_dataset[0]) == 3:
        use_built_in_rpn = True
    else:
        use_built_in_rpn = False

    collate_fn = rpn_collate_fn(use_built_in_rpn)

    roi_dataset = RoiHeadsDataset(device)
    data_loader = DataLoader(rpn_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
                  
    for idx, args in enumerate(data_loader):
        feature_map = args[-2]
        proposals = rpn(*args)[0][0]
        image_sizes = dataset[idx][0].shape[-2:]
        targets = dataset[idx][1]
        roi_dataset.append((feature_map, proposals, image_sizes, targets))

    return roi_dataset


class roi_collate_fn:
    def __init__(self, use_built_in_roi_heads=False):
        if not use_built_in_roi_heads:
            print("WARNING: Custom ROI Heads Network has not been fully integrated")
        self._collate_fn = self._builtin_fn

    def __call__(self, data):
        return self._collate_fn(data)
        
    def _builtin_fn(self, data):
        features    = []
        proposals   = []
        image_sizes = []
        targets     = []
        for sample in data:
            features.append(sample[0])
            proposals.append(sample[1])
            image_sizes.append(sample[2])
            targets.append(sample[3])
        features = {'0' : torch.cat(features)}
        return features, proposals, image_sizes, targets


class roi_loss_fn:
    def __init__(self, weights, use_built_in_roi_heads=False):
        if not use_built_in_roi_heads:
            print("WARNING: Custom ROI Heads Network has not been fully integrated")
        self._loss_fn = self._builtin_fn
        self._weights = weights

    def __call__(self, model_output):
        return self._loss_fn(model_output)
    
    def _builtin_fn(self, model_output):
        loss_dict = model_output[1]
        losses = 0
        for key, loss in loss_dict.items():
            losses = losses + loss * self._weights[key]
        return losses


def roi_log_fn(loss, model_outputs):
    loss_dict = model_outputs[1]
    return {'Loss/train'            : loss,
            'Classifier_Loss/train' : loss_dict['loss_classifier'],
            'Box_Loss/train'        : loss_dict['loss_box_reg']}


if __name__ == "__main__":

    torch.manual_seed(69)
    torch.autograd.set_detect_anomaly(True)


    # Object
    # obj = classifierNet(10700).cuda()
    # obj = classifierNet(48).cuda()
    obj = headNet(32)
    obj.initDataLoader()


    # obj.load_state_dict(torch.load("model_20240502_134909_1", map_location=torch.device('cpu')))
    # obj.load_state_dict(torch.load("model_20240502_180252_0", map_location=torch.device('cpu')))
    # obj.load_state_dict(torch.load("model_20240502_194941_3", map_location=torch.device('cpu')))
    # obj.load_state_dict(torch.load("model_20240503_152227_4", map_location=torch.device('cpu')))
    # Best so far: model_20240503_014417_17 (old network) model_20240503_141302_0 (new network)

    # Run training
    # obj.runTraining(num_epochs=1000)

    # Test results
    obj.testClassifier()