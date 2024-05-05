from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from SummaryWriter import SummaryWriter
from ClassConstants import ClassConstants
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from datetime import datetime
from DataManager import DataManager
from PathConstants import PathConstants
from FlirDataset import FlirDataset
import numpy as np
import torch
import random
import math
import os
import re

torch.autograd.set_detect_anomaly(True)

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

def loss_fn(loss_dict):
    losses = sum(loss for loss in loss_dict.values())
    return losses

def default_loss_fn(model_output):
    return model_output

def default_log_fn(loss, _):
    return {'Loss/train' : loss}

class CustomSampler(Sampler):
    def __init__(self, indices=None):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

class NetworkTrainer:

    # Class constructor
    def __init__(self,
                 data,
                 model,
                 optimizer,
                 run_folder,
                 num_epochs=1,
                 batch_size=1,
                 collate_fn=None,
                 loss_fn=None,
                 log_fn=None,
                 save_period={'epoch': 1, 'batch':-1},
                 device=torch.device('cpu')):
        
        # Save inputs to class constructor
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.loss_fn = loss_fn
        self.log_fn = log_fn
        self.device = device
        self.save_period = save_period

        # Determine the number of batches
        self.num_batches = math.ceil(len(self.data)/self.batch_size)

        # Determine loss function
        if self.loss_fn is None:
            self.loss_fn = default_loss_fn

        # Determnine log funciton
        if self.log_fn is None:
            self.log_fn = default_log_fn

        # Initial epoch and batch number
        self.epoch = 0
        self.batch = 0

        # Initial indices for sampler
        self.sampler_indices = None

        # Initial states for the random number generator
        self.init_rng_state = None
        self.epoch_rng_state = None
        self.rng_state = None

        # Initial set of loss values
        self.loss = []

        # Create run directory for outputs
        self.get_run_dir(run_folder)
        os.makedirs(self.run_dir)

        # Create TensorBoard SummaryWriter instance
        self.summary_writer = SummaryWriter(self.run_dir)
        
    # Function gets the initial state of the random number generator
    def get_init_rng_state(self):
        self.init_rng_state = {
            'numpy'  : np.random.get_state(),
            'torch'  : torch.random.get_rng_state(),
            'random' : random.getstate()}

    # Function gets the epoch state of the random number generator
    def get_epoch_rng_state(self):
        self.epoch_rng_state = {
            'numpy'  : np.random.get_state(),
            'torch'  : torch.random.get_rng_state(),
            'random' : random.getstate()}
        
    # Function gets the current state of the random number generator
    def get_rng_state(self):
        self.rng_state = {
            'numpy'  : np.random.get_state(),
            'torch'  : torch.random.get_rng_state(),
            'random' : random.getstate()}

    # Function loads the initial random number generator state
    def load_init_rng_state(self):
        np.random.set_state(self.init_rng_state['numpy'])
        torch.random.set_rng_state(self.init_rng_state['torch'])
        random.setstate(self.init_rng_state['random'])

    # Function loads the random number generator state from the last epoch
    def load_epoch_rng_state(self):
        np.random.set_state(self.epoch_rng_state['numpy'])
        torch.random.set_rng_state(self.epoch_rng_state['torch'])
        random.setstate(self.epoch_rng_state['random'])

    # Function loads the random number generator state
    def load_rng_state(self):
        np.random.set_state(self.rng_state['numpy'])
        torch.random.set_rng_state(self.rng_state['torch'])
        random.setstate(self.rng_state['random'])

    # Function saves the training state
    def save_state(self):

        # Get the current state of the random number generator
        self.get_rng_state()

        # Determine what to name the checkpoint file
        checkpoint_file = f"cp__epoch_{self.epoch}_batch_{self.batch}.pth"
        checkpoint_file = os.path.join(self.run_dir, checkpoint_file)

        # Save the checkpoint file
        torch.save({
            'epoch'           : self.epoch,
            'batch'           : self.batch,
            'loss'            : self.loss,
            'indices'         : self.sampler_indices,
            'model_state'     : self.model.state_dict(),
            'optimizer_state' : self.optimizer.state_dict(),
            'sampler_indices' : self.sampler_indices,
            'rng_state'       : self.rng_state,
            'init_rng_state'  : self.init_rng_state}, checkpoint_file)
        
        # Get a list of all files in the run directory
        files = os.listdir(self.run_dir)

        # Determine which files are checkpoint files
        r = re.compile('cp.*\\.pth')
        files = list(filter(r.match, files))

        # Delete all the old checkpoint files
        checkpoint_file = os.path.basename(checkpoint_file)
        for file in files:
            if file != checkpoint_file:
                try:
                    file = os.path.join(self.run_dir,file)
                    os.remove(file)
                except:
                    print("Failed to remove checkpoint file %s" % (file))

    # Function parses a run directory
    def parse_run_dir(self, run_dir):
        
        # Get contents of run directory
        dir_contents = os.listdir(run_dir)

        # Get all files in run directory
        file_names = []
        for name in dir_contents:
            name = os.path.join(run_dir, name)
            if os.path.isfile(name):
                file_names.append(os.path.basename(name))

        # Match all checkpoint files
        r = re.compile('^cp__epoch_(\\d+)_batch_(\\d+)\\.pth$')
        matches = []
        for file_name in file_names:
            m = r.match(file_name)
            if m is not None:
                matches.append(m)

        # Extra meta data from log file name
        checkpoint_files = [m.group(0) for m in matches]
        epoch = np.array([int(m.group(1)) for m in matches])
        batch = np.array([int(m.group(2)) for m in matches])

        # Get the latest checkpoint file and the corresponding loop iteration
        num_iter = batch + epoch * self.num_batches
        max_iter = num_iter.max()
        max_idx = np.where(num_iter == max_iter)[0][0]
        checkpoint_file = checkpoint_files[max_idx]
        checkpoint_file = os.path.join(run_dir, checkpoint_file)

        # Get the tensor board event file
        r = re.compile('^events.out.tfevents.*$')
        matches = []
        for file_name in file_names:
            m = r.match(file_name)
            if m is not None:
                matches.append(m)
        event_file = matches[0].group(0)
        event_file = os.path.join(run_dir, event_file)

        return checkpoint_file, event_file, max_iter

    # Function loads the training state
    def load_state(self, run_dir):

        # Get checkpoint file and event file from run directory
        checkpoint_file, event_file, num_events = self.parse_run_dir(run_dir)

        # Add events to tensor board
        self.summary_writer.load_state(event_file, num_events)

        # Load the checkpoint file
        state = torch.load(checkpoint_file)

        # Set the epoch and batch number
        self.epoch = state['epoch']
        self.batch = state['batch']

        # Load the model and optimizer state
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])

        # Load the set of indices used by the sampler
        self.sampler_indices = state['sampler_indices']

        # Get the state of the random number generator
        self.rng_state = state['rng_state']
        self.init_rng_state = state['init_rng_state']

        # Load the initial random number generator state
        # State at the start of the last epoch
        self.load_init_rng_state()
         
    # Function computes the path to the run directory
    def get_run_dir(self, run_folder):
        class_path = os.path.dirname(__file__)
        run_root_dir = os.path.join(class_path,'..','runs',run_folder)
        run_root_dir = os.path.abspath(run_root_dir)
        current_time = datetime.now()
        current_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(run_root_dir,f"run__{current_time}")
        
    # Function defines the training loop
    def train(self):

        # Save the initial random generator state
        self.get_init_rng_state()

        # Create custom sampler
        sampler = CustomSampler()

        # Create data loader object
        data_loader = DataLoader(self.data, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=False, sampler=sampler)

        # Put the model in training model
        self.model.train()

        # Specify whether random number generator state must
        # be set before first epoch
        set_epoch_rng_state = self.epoch_rng_state is not None

        # Initialize counters for the number of batches and the number of epochs 
        batch_count = 0
        epoch_count = 0

        # Determine initial epoch
        init_epoch = self.epoch

        # Loop for each epoch
        for self.epoch in range(init_epoch, self.num_epochs):

            # Print the current epoch number
            print(f'Beginning epoch {self.epoch+1}/{self.num_epochs}:')

            # Set random number generator state for first epoch
            if set_epoch_rng_state:
                set_epoch_rng_state = False
                self.load_epoch_rng_state()

            # Saving state from last batch or last epoch
            if ((epoch_count == self.save_period['epoch']) or
                ((batch_count > 0) and (self.save_period['batch'] >= 0))):
                self.save_state()

            # Reset epoch counter after saving epoch data
            if (epoch_count == self.save_period['epoch']):
                epoch_count = 0
            
            # Reset batch counter
            batch_count = 0

            # Assume the batch random number generator state needs to be loaded
            load_rng_state = True

            # If the batch number is zero, we are starting a fresh epoch
            if self.batch == 0:

                # Set the initial indices for the sampler
                self.sampler_indices = np.random.permutation(len(self.data))

                # Don't load the state of the random number generator
                load_rng_state = False

            # Set the indices for the custom sampler
            sampler.indices = self.sampler_indices[self.batch * self.batch_size::]

            # Save the random generator state at the start of the epoch
            self.get_epoch_rng_state()

            # Loop for each batch
            for args in data_loader:

                # Load the random number generator state if resuming training mid epoch
                if load_rng_state:
                    self.load_rng_state()
                    load_rng_state = False

                # Save training state from end of last batch
                if (batch_count == self.save_period['batch']):
                    self.save_state()
                    batch_count = 0

                # Set the gradient to zero
                self.optimizer.zero_grad()

                # Compute the total loss
                model_output = self.model(*args)
                loss = self.loss_fn(model_output)

                # Perform backprogation
                loss.backward()
                self.optimizer.step()

                # Log the loss to TensorBoard
                log_dict = self.log_fn(loss, model_output)
                idx = self.batch + self.epoch*self.num_batches
                for key, value in log_dict.items():
                    self.summary_writer.add_scalar(key, value.item(), idx)
                self.summary_writer.flush()

                # Print the batch loss
                print(f'Batch Loss ({self.batch+1}/{self.num_batches}): {loss.item()}')

                # Append to array of losses
                self.loss.append(loss.item())

                # Increment the batch counters
                self.batch += 1
                batch_count += 1

            # Reset the batch number
            self.batch = 0

            # Increment the epoch counter
            epoch_count += 1

        # Save data from the end of the final epoch
        self.epoch = self.num_epochs
        self.save_state()

        # Clear any pending events
        self.summary_writer.flush()
        self.summary_writer.close()
      
# Code to run if file is called directly
if __name__ == "__main__":

    # Download data and create path constants singleton
    data_manager = DataManager('train')
    data_manager.download_datasets()
    data_dir = data_manager.get_download_dir()
    PathConstants(data_dir)

    # Set the initial random number generator seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    # Determine the device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Create a faster RCNN model
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(ClassConstants.LABELS.keys()))
    model.to(device)

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0)

    # Set the period for saving data
    # -1 will cause data not to be saved
    save_period = {'epoch' : 1, 'batch' : 1}

    # Create dataset object
    train_data = FlirDataset(PathConstants.TRAIN_DIR, downsample=1, num_images=10, device=device)

    # Run subfolder
    run_folder = 'built_in_faster_rcnn'

    # Create network trainer
    net_trainer = NetworkTrainer(
        data        = train_data, 
        model       = model,
        optimizer   = optimizer,
        run_folder  = run_folder,
        num_epochs  = 50,
        batch_size  = 1,
        loss_fn     = loss_fn,
        collate_fn  = collate_fn,
        save_period = save_period,
        device      = device
    )

    # Uncomment and adjust path to resume training
    # net_trainer.load_state(os.path.join(os.path.dirname(__file__),'run','run__2024-05-04_14-52-14'))
    
    # Train model
    net_trainer.train()
