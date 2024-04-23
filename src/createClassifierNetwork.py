import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from FlirDataset import FlirDataset
from PathConstants import PathConstants


class classifierNet(nn.module):


    def __init__(self):
        super(classifierNet, self).__init__()

        # First fully connected layer
        self.fc1 = nn.Linear(256, 1024, bias=True)

        # Second fully connected layer
        self.fc2 = nn.Linear(512, 2048, bias=True)

        # Third fully connected layer
        self.fc3 = nn.Linear(1024, 1024, bias=True)

        # Fourth fully connected layer
        self.fc4 = nn.Linear(512, 512, bias=True)

        # Create data loaders for our datasets; shuffle for training, not for validation
        self.training_set = FlirDataset(PathConstants.TRAIN_DIR)
        self.validation_set = FlirDataset(PathConstants.VAL_DIR)
        self.training_loader = torch.utils.data.DataLoader(self.training_set, batch_size=4, shuffle=False)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_set, batch_size=4, shuffle=False)
        print('Training set has {} instances'.format(len(self.training_set)))
        print('Validation set has {} instances'.format(len(self.validation_set)))

        # Not sure what this does
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
        
        # Class labels
        self.labels = ('Car', 'Truck', 'Scooter')

        # Specify loss function
        self.loss_function = nn.MSEloss()


    def forward(self, x):

        # Pass input through layers, with relu and max pooling nonlinearities
        x = self.fc1(x)                     # 256x256 -> 1024x1024
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 1024x1024 -> 512x512

        x = self.fc2(x)                     # 512x512 -> 2048x2048
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 2048x2048 -> 1024x1024

        x = self.fc3(x)                     # 1024x1024 -> 1024x1024
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 1024x1024 -> 512x512

        x = self.fc4(x)                     # 512x512 -> 512x512
        x = F.relu(x)
        x = F.max_pool1d(x, 2, stride=2)    # 512x512 -> 256x256

        # Softmax
        output = F.log_softmax(x, dim=1)
        return output
    

    def train_one_epoch(self, epoch_index, tb_writer):

        # Specify optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        # Train on data
        for i, data in enumerate(self.training_loader):
           
            # Load data
            inputs, labels = data

            # Zero gradients before each batch
            optimizer.zero_grad()

            # Run batch through network
            outputs = self(inputs)

            # Calculate loss and gradients
            loss = self.loss_function(outputs, labels)
            loss.backward()
            
            # Adjust weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                print(' batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0
            
        return last_loss
    

    def pre_epoch_activity(self, num_epochs=5):

        # Initialize params
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0
        best_vloss = 1_000_000.

        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Track gradient
            self.train(True)
            avg_loss = self.train_one_poch(epoch_number, writer)

            running_vloss = 0.0
            self.eval()

            # Disable gradient computation to save mem
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    voutputs = self(vinputs)
                    vloss = self.loss_function(voutputs, vlabels)
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