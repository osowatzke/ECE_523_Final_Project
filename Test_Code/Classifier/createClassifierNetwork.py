import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from FlirDataset import FlirDataset
from PathConstants import PathConstants
import loadTrainingData as LTD
from torch.utils.data.sampler import Sampler
from BackboneNetwork import BackboneNetwork
import numpy as np
from ClassConstants import ClassConstants
import math

device = torch.device('cuda')
torch.cuda.set_device(0)

IMG_HEIGHT = 7
IMG_WIDTH = 7
IMG_CHANNELS = 2048

BATCH_SIZE = 24

BACKBONE_FLAG = True    # True = run data through backbone before classifier


class classifierNet(nn.Module):


    def __init__(self, num_images_in):
        super(classifierNet, self).__init__()

        # Class labels
        classes = ClassConstants()
        self.labels = classes.LABELS

        # First fully connected layer
        self.fc1 = nn.Linear(IMG_HEIGHT*IMG_WIDTH*IMG_CHANNELS, 256, bias=True)

        # Second fully connected layer
        self.fc2 = nn.Linear(128, 256, bias=True)

        # Third fully connected layer
        self.fc3 = nn.Linear(128, 256, bias=True)

        # Batch normalization
        self.m = nn.BatchNorm1d(128)

        # Fourth fully connected layer
        self.fc4 = nn.Linear(128, 2*len(self.labels), bias=True)

        # More batch norm
        self.mm = nn.BatchNorm1d(len(self.labels))

        # Softmax
        self.softmax = torch.nn.Softmax(dim = 1)

        # Initialize weights
        # (already done by Pytorch automatically)

        # Create data loaders for our datasets; shuffle for training, not for validation
        self.numSamples = num_images_in
        PathConstants()
        sampler = LTD.CustomSampler()
        valid_sampler = LTD.CustomSampler()
        # sampler_indices = np.random.permutation(num_images_in)
        sampler_indices = range(int(num_images_in + BATCH_SIZE % num_images_in))
        sampler.indices = sampler_indices
        valid_sampler.indices = range(int(num_images_in/1000 + BATCH_SIZE % (num_images_in/1000)))
        self.training_set = FlirDataset(PathConstants.TRAIN_DIR, num_images=int(num_images_in + BATCH_SIZE % num_images_in), downsample=1, device=None)
        # self.validation_set = FlirDataset(PathConstants.VAL_DIR, num_images=int(num_images_in/10) + int((num_images_in/10)) % BATCH_SIZE, downsample=1, device=None)
        self.validation_set = FlirDataset(PathConstants.VAL_DIR, num_images=int(num_images_in/1000 + BATCH_SIZE % (num_images_in/1000)), downsample=1, device=None)
        self.training_loader = torch.utils.data.DataLoader(self.training_set, batch_size=BATCH_SIZE, collate_fn=LTD.collate_fn, shuffle=False, sampler=sampler)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_set, batch_size=BATCH_SIZE, collate_fn=LTD.collate_fn, shuffle=False, sampler=valid_sampler)
        print('Training set has {} instances'.format(len(self.training_set)))
        print('Validation set has {} instances'.format(len(self.validation_set)))

        # Not sure what this does
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        # Specify loss function
        # self.loss_function = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        self.loss_function = nn.CrossEntropyLoss()


    def forward(self, x):

        # Pass input through layers, with relu and max pooling nonlinearities
        # x = x.double()
        x = self.fc1(x)                     # ??? -> 256
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 256 -> 128

        x = self.fc2(x)                     # 128 -> 256
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 256 -> 128

        # x = self.fc3(x)                     # 128 -> 256
        # x = F.relu(x)
        # x = F.max_pool1d(x, 2, stride=2)    # 256 -> 128
        
        x = self.m(x)

        x = self.fc4(x)                     # 128 -> 32
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 32 -> 16

        # x = self.mm(x)

        # output = x

        # Softmax
        output = F.log_softmax(x, dim=1)
        # output = self.softmax(x)
        
        return output
    

    def trainOneEpoch(self, epoch_index, tb_writer, backbone=None):

        # Loss metrics
        running_loss = 0.
        last_loss = 0.

        # Specify optimizer
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, eps=1e-08)

        # For each batch
        for i, data in enumerate(self.training_loader):
           
            # Load data
            inputs, labels = data

            # Run data through backbone
            if backbone:
                imgs = torch.stack((*inputs,))
                # features = torch.tensor(backbone(imgs))
                features = (backbone(imgs+.0001))

            # Number of ROIs (removing ROIs with class -1)
            numROIs = 0
            for j in range(len(labels)):
                for p in range(len(labels[j]["labels"])):
                    numROIs = numROIs + int(labels[j]["labels"][p] != -1.)

            # Hold all data
            # labels_ = np.empty((numROIs, len(self.labels)))
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
                        # vec_ = torch.zeros(len(self.labels))
                        # vec_[int(class_)] = 1.0
                        labels_[totalROIs, int(class_)] = 1.0
                        classes_[totalROIs] = int(class_)

                        # Extracting bounding boxes
                        boxes_[totalROIs, :] = label["boxes"][j]
                        temp[numTrueROIs,:] = label["boxes"][j]
                        boxes_db[totalROIs] = torch.cat((torch.tensor([k]).to(device), (label["boxes"][j]).to(device)), 0)

                        # Extracting ROIs
                        # x1 = abs(math.floor(boxes_[k+j, 0] / 32))
                        # y1 = abs(math.floor(boxes_[k+j, 1] / 32))
                        # x2 = abs(math.ceil(boxes_[k+j, 2]+1 / 32))
                        # y2 = abs(math.ceil(boxes_[k+j, 3]+1 / 32))
                        # img_ = features[k,:,y1:y2,x1:x2]
                        # img_ = (img_ - img_.min())/(img_.max() - img_.min())    # Image normalization
                        # imgs_[k+numTrueROIs,:,:,:] = F.interpolate(img_[None,:,:,:], size=(7, 7))

                        numTrueROIs = numTrueROIs + 1
                        totalROIs = totalROIs + 1
                k = k + 1
                # boxes_db.append(temp)
            
            # Experimental: Pytorch RoI Pooling
            imgs_pyt = torchvision.ops.roi_pool(features.to(device), boxes_db.to(device), (7, 7))

            # Re-format feature space to make it compatible with FCL dimensions
            features = torch.flatten(imgs_pyt, start_dim = 1)

            # Zero gradients before each batch
            optimizer.zero_grad()

            # Run batch through network
            outputs = self(features)

            # Calculate loss and gradients
            classes_ = classes_.type(torch.LongTensor).to(device)
            loss = self.loss_function(outputs, classes_)
            loss.backward()

            # Clip gradient
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)

            # DEBUG# for name, param in self.named_parameters():
            #     print(name, torch.isfinite(param.grad).all())
            
            # Adjust weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 10 == 0:
                last_loss = running_loss / 10
                print(' batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * self.numSamples + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0
            
        return last_loss
    

    def runTraining(self, num_epochs=5):

        # Initialize params
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
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

                    # Load data
                    vinputs, vlabels = vdata

                    # Run data through backbone
                    if backbone:
                        vimgs = torch.stack((*vinputs,))
                        # features = torch.tensor(backbone(vimgs))
                        vfeatures = (backbone(vimgs))

                    # Number of ROIs (removing ROIs with class -1)
                    vnumROIs = 0
                    for j in range(len(vlabels)):
                        for p in range(len(vlabels[j]["labels"])):
                            vnumROIs = vnumROIs + int(vlabels[j]["labels"][p] != -1.)

                    # Hold all data
                    vlabels_ = torch.zeros([vnumROIs, len(self.labels)]).to(device)
                    vclasses_ = torch.zeros([vnumROIs]).to(device)
                    vimgs_ = torch.zeros([vnumROIs, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH]).to(device)
                    vboxes_ = torch.zeros([vnumROIs, 4]).to(device)
                    vboxes_db = torch.zeros([vnumROIs, 5]).to(device)

                    # ROI Pooling
                    k = 0
                    vtotalROIs = 0
                    for vlabel in vlabels:

                        # Number of ROIs in this image
                        vnumROIs_this_img = 0
                        for p in range(len(vlabel["labels"])):
                            vnumROIs_this_img = vnumROIs_this_img + int(vlabel["labels"][p] != -1.)
                    
                        # For each ROI in image
                        j = 0
                        vnumTrueROIs = 0
                        vtemp = torch.zeros(vnumROIs_this_img, 4).to(device)
                        for j in range(len(vlabel["labels"])):

                            # Determine class
                            vclass_ = vlabel["labels"][j]

                            # If it's not class -1:
                            if vclass_ != torch.tensor(-1.).to(device):

                                # Creating ideal output
                                vlabels_[vtotalROIs, int(vclass_)] = 1.0
                                vclasses_[vtotalROIs] = int(vclass_)

                                # Extracting bounding boxes
                                vboxes_[vtotalROIs, :] = vlabel["boxes"][j]
                                vtemp[vnumTrueROIs,:] = vlabel["boxes"][j]
                                vboxes_db[vtotalROIs] = torch.cat((torch.tensor([k]).to(device), (vlabel["boxes"][j]).to(device)), 0)

                                # Extracting ROIs
                                # x1 = abs(math.floor(vboxes_[k+j, 0] / 32))
                                # y1 = abs(math.floor(vboxes_[k+j, 1] / 32))
                                # x2 = abs(math.ceil(vboxes_[k+j, 2]+1 / 32))
                                # y2 = abs(math.ceil(vboxes_[k+j, 3]+1 / 32))
                                # vimg_ = vfeatures[k,:,y1:y2,x1:x2]
                                # vimgs_[k+vnumTrueROIs,:,:,:] = F.interpolate(vimg_[None,:,:,:], size=(7, 7))

                                vnumTrueROIs = vnumTrueROIs + 1
                                vtotalROIs = vtotalROIs + 1
                        k = k + 1
                        # vboxes_db.append(vtemp)
            
                    # Experimental: Pytorch RoI Pooling
                    vimgs_pyt = torchvision.ops.roi_pool(vfeatures.to(device), vboxes_db.to(device), (7, 7))

                    # Re-format feature space to make it compatible with FCL dimensions
                    vfeatures = torch.flatten(vimgs_pyt, start_dim = 1)

                    # Run batch through network
                    voutputs = self(vfeatures)

                    # Calculate loss and gradients
                    vclasses_ = vclasses_.type(torch.LongTensor).to(device)
                    vloss = self.loss_function(voutputs, vclasses_)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

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


if __name__ == "__main__":

    torch.manual_seed(69)
    torch.autograd.set_detect_anomaly(True)

    # Object
    obj = classifierNet(10700).cuda()
    obj.load_state_dict(torch.load("model_20240502_115006_7"))

    # Run training
    obj.runTraining(num_epochs=1000)