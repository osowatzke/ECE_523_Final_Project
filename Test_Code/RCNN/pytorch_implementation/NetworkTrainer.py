from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ClassConstants import ClassConstants
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from datetime import datetime
from PathConstants import PathConstants
from FlirDataset import FlirDataset
import numpy as np
import torch
import random
import math
import os

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

class CustomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

class NetworkTrainer:

    def __init__(self, data, model, optimizer, num_epochs=1, batch_size=1, save_period={'epoch': 1, 'batch':-1}, device=torch.device('cpu')):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.save_period = save_period
        self.epoch = 0
        self.batch = 0
        self.sampler_indices = None
        self.loss = []

        self.get_run_dir()
        os.makedirs(self.run_dir)
        
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)

    def get_rng_state(self):
        self.rng_state = {
            'numpy'  : np.random.get_state(),
            'torch'  : torch.random.get_rng_state(),
            'random' : random.getstate()}

    def load_rng_state(self):
        np.random.set_state(self.rng_state['numpy'])
        torch.random.set_rng_state(self.rng_state['torch'])
        random.setstate(self.rng_state['random'])

    def save_state(self):
        self.get_rng_state()
        current_time = datetime.now()
        current_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_file = f"cp__{current_time}.pth"
        checkpoint_file = os.path.join(self.run_dir, checkpoint_file)
        torch.save({
            'epoch'           : self.epoch,
            'batch'           : self.batch,
            'loss'            : self.loss,
            'indices'         : self.sampler_indices,
            'model_state'     : self.model.state_dict(),
            'optimizer_state' : self.optimizer.state_dict(),
            'sampler_indices' : self.sampler_indices,
            'rng_state'       : self.rng_state}, checkpoint_file)
    
    def load_state(self, checkpoint):
        state = torch.load(checkpoint)
        self.epoch = state['epoch']
        self.batch = state['batch']
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.sampler_indices = state['sampler_indices']
        self.rng_state = state['rng_state']
        self.load_rng_state()
         
    def get_run_dir(self):
        class_path = os.path.dirname(__file__)
        current_time = datetime.now()
        current_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(class_path,'run',f"run__{current_time}")
        
    def train(self):
        num_batches = math.ceil(len(self.data)/self.batch_size)
        sampler = CustomSampler(None)
        data_loader = DataLoader(self.data, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=False, generator=None, sampler=sampler)
        self.model.train()
        epoch_count = 0
        while (self.epoch < self.num_epochs):
            print(f'Beginning epoch {self.epoch+1}/{self.num_epochs}:')
            load_rng_state = True
            if self.batch == 0:
                self.sampler_indices = np.random.permutation(len(self.data))
                load_rng_state = False
            sampler.indices = self.sampler_indices[self.batch * self.batch_size::]
            batch_count = 0
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            for img, targets in data_loader:
                print(img[0].shape)
                if load_rng_state:
                    self.load_rng_state()
                    load_rng_state = False
                self.optimizer.zero_grad()
                loss_dict = self.model(img, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()
                self.batch += 1
                batch_count += 1
                print(f'Batch Loss ({self.batch}/{num_batches}): {losses.item()}')
                self.loss.append(losses.item())
                if (self.batch == num_batches):
                    break
                elif (batch_count == self.save_period['batch']):
                    self.save_state()
                    batch_count = 0
            self.batch = 0
            epoch_count += 1     
            self.epoch += 1
            if (epoch_count == self.save['epoch']):
                self.save()
                epoch_count = 0
                
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--make_local_copy', action='store_true')
    args = parser.parse_args()
    PathConstants = PathConstants()
    if args.make_local_copy:
        PathConstants.source_from_local_copy()

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(ClassConstants.LABELS.keys()))
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7, momentum=0)

    save_period = {'epoch' : 1, 'batch' : 100}

    train_data = FlirDataset(PathConstants.TRAIN_DIR, device=device)

    net_trainer = NetworkTrainer(
        data        = train_data, 
        model       = model,
        optimizer   = optimizer,
        num_epochs  = 1,
        batch_size  = 16,
        save_period = save_period,
        device      = device
    )

    # net_trainer.load_state(os.path.join(os.path.dirname(__file__),'run__2024-04-24_01-27-11','cp__2024-04-24_01-27-22.pth'))
    net_trainer.train()
