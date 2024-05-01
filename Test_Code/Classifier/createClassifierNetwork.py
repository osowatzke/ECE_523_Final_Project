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

IMG_HEIGHT = 20
IMG_WIDTH = 16
IMG_CHANNELS = 2048

BACKBONE_FLAG = True    # True = run data through backbone before classifier


class classifierNet(nn.Module):


    def __init__(self, num_images_in):
        super(classifierNet, self).__init__()

        # Class labels
        classes = ClassConstants()
        self.labels = classes.LABELS

        # First fully connected layer
        self.fc1 = nn.Linear(IMG_HEIGHT*IMG_WIDTH*IMG_CHANNELS, 1024, bias=True)

        # Second fully connected layer
        self.fc2 = nn.Linear(512, 2048, bias=True)

        # Third fully connected layer
        self.fc3 = nn.Linear(1024, 1024, bias=True)

        # Fourth fully connected layer
        self.fc4 = nn.Linear(512, 2*len(self.labels), bias=True)

        # Create data loaders for our datasets; shuffle for training, not for validation
        self.numSamples = num_images_in
        PathConstants()
        sampler = LTD.CustomSampler()
        sampler_indices = np.random.permutation(num_images_in)
        sampler.indices = sampler_indices
        self.training_set = FlirDataset(PathConstants.TRAIN_DIR, num_images=num_images_in, downsample=1, device=None)
        self.validation_set = FlirDataset(PathConstants.VAL_DIR, num_images=int(num_images_in/10), downsample=1, device=None)
        self.training_loader = torch.utils.data.DataLoader(self.training_set, batch_size=12, collate_fn=LTD.collate_fn, shuffle=False, sampler=sampler)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_set, batch_size=12, collate_fn=LTD.collate_fn, shuffle=False, sampler=sampler)
        print('Training set has {} instances'.format(len(self.training_set)))
        print('Validation set has {} instances'.format(len(self.validation_set)))

        # Not sure what this does
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

        # Specify loss function
        self.loss_function = nn.MSELoss()


    def forward(self, x):

        # Pass input through layers, with relu and max pooling nonlinearities
        x = x.float()
        x = self.fc1(x)                     # ??? -> 1024
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 1024 -> 512

        x = self.fc2(x)                     # 512 -> 2048
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 2048 -> 1024

        x = self.fc3(x)                     # 1024 -> 1024
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 1024 -> 512

        x = self.fc4(x)                     # 512 -> 512
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 512 -> 256

        self.float()

        # Softmax
        output = F.log_softmax(x, dim=1)
        
        return output.float()
    

    def trainOneEpoch(self, epoch_index, tb_writer, backbone=None):

        # Loss metrics
        running_loss = 0.
        last_loss = 0.

        # Specify optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        # Train on data
        for i, data in enumerate(self.training_loader):
           
            # Load data
            inputs, labels = data

            # Reformat labels
            labels_ = np.empty((12, len(self.labels)))
            j = 0
            for label in labels:
                class_ = label["labels"][0]
                vec_ = torch.zeros(len(self.labels))
                vec_[int(class_)] = torch.tensor(1.0)
                labels_[j,:] = vec_
                j = j + 1

            # Run data through backbone
            if backbone:
                imgs = torch.stack((*inputs,))
                features = torch.tensor(backbone(imgs))

            # Re-format to make it compatible with FCL dimensions
            features = torch.flatten(features, start_dim = 1)

            # Zero gradients before each batch
            optimizer.zero_grad()

            # Run batch through network
            outputs = self(features)

            # Calculate loss and gradients
            loss = self.loss_function(outputs, torch.tensor(labels_).float())
            loss.backward()
            
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

                    vinputs, vlabels = vdata

                    # Reformat labels
                    vlabels_ = np.empty((12, len(self.labels)))
                    j = 0
                    for vlabel in vlabels:
                        vclass_ = vlabel["labels"][0]
                        vvec_ = torch.zeros(len(self.labels))
                        vvec_[int(vclass_)] = torch.tensor(1.0)
                        vlabels_[j,:] = vvec_
                        j = j + 1

                    # Run data through backbone
                    if backbone:
                        vimgs = torch.stack((*vinputs,))
                        vfeatures = torch.tensor(backbone(vimgs))

                    # Re-format to make it compatible with FCL dimensions
                    vfeatures = torch.flatten(vfeatures, start_dim = 1)

                    # Run through network
                    voutputs = self(vfeatures)
                    vloss = self.loss_function(voutputs, torch.tensor(vlabels_).float())
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

    # Object
    obj = classifierNet(1200) # Number of images MUST be a multiple of batch size

    # Run training
    obj.runTraining(num_epochs=10)