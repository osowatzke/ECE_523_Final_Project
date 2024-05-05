from BackboneNetwork        import BackboneNetwork
from DataManager            import DataManager
from NetworkTrainer         import NetworkTrainer
from PathConstants          import PathConstants
from RegionProposalNetwork  import *

import argparse

# Parse optional input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--loss_weights', nargs=2, default=[1,1], type=float)
parser.add_argument('-l', '--learning_rate', default=0.01, type=float)

# Flags to switch between built-in and custom implementations
use_built_in_rpn = False

# Create Data Manager Object
# Will download dataset if not already available
data_manager = DataManager()
data_manager.download_datasets()

# Create Path Constants Singleton
data_dir = data_manager.get_download_dir()
PathConstants(data_dir)

# Create input dataset
dataset = FlirDataset(PathConstants.TRAIN_DIR)

# Create backbone network
backbone = BackboneNetwork()

# Create RPN dataset
rpn_dataset = create_rpn_dataset(backbone, dataset, use_built_in_rpn)

# Get sizes of images and feature maps
image_size = dataset[0][0].shape
if use_built_in_rpn:
    feature_map_size = rpn_dataset[0][1].shape
else:
    feature_map_size = rpn_dataset[0][0].shape

# Create RPN
torch.manual_seed(0)
rpn = create_region_proposal_network(image_size, feature_map_size, use_built_in_rpn)
rpn.train()

# Train the RPN network
optimizer = torch.optim.SGD(
    rpn.parameters(),
    lr = parser.learning_rate,
    momentum = 0.9,
    weight_decay = 5e-3)

# Create loss function with user weights
weights = {
    "loss_objectness"  : parser.loss_weights[0], 
    "loss_rpn_box_reg" : parser.loss_weights[1]}
loss_fn = rpn_loss_fn(weights)

collate_fn = rpn_collate_fn(use_built_in_rpn)

run_folder = 'region_proposal_network'

network_trainer = NetworkTrainer(
    data       = rpn_dataset,
    model      = rpn,
    optimizer  = optimizer,
    run_folder = run_folder,
    num_epochs = 50,
    batch_size = 128,
    loss_fn    = loss_fn,
    collate_fn = collate_fn)

network_trainer.train() 
