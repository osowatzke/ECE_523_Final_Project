from BackboneNetwork        import BackboneNetwork
from DataManager            import DataManager
from FlirDataset            import FlirDataset
from NetworkTrainer         import NetworkTrainer
from PathConstants          import PathConstants
from RegionProposalNetwork  import *

import argparse

# Parse optional input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--loss_weights', nargs=2, default=[1,1], type=float)
parser.add_argument('-l', '--learning_rate', default=0.01, type=float)
args = parser.parse_args()

# Determine the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

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
dataset = FlirDataset(PathConstants.TRAIN_DIR, device=device)

# Create backbone network
backbone = BackboneNetwork()
backbone.to(device)

# Create RPN dataset
rpn_dataset = create_rpn_dataset(backbone, dataset, use_built_in_rpn, device)

# Get sizes of images and feature maps
image_size = dataset[0][0].shape
if use_built_in_rpn:
    feature_map_size = rpn_dataset[0][1].shape
else:
    feature_map_size = rpn_dataset[0][0].shape

# Create RPN
torch.manual_seed(0)
rpn = create_region_proposal_network(image_size, feature_map_size, use_built_in_rpn)
rpn.to(device)
rpn.train()

# Train the RPN network
optimizer = torch.optim.SGD(
    rpn.parameters(),
    lr = args.learning_rate,
    momentum = 0.9,
    weight_decay = 5e-3)

# Create loss function with user weights
weights = {
    "loss_objectness"  : args.loss_weights[0], 
    "loss_rpn_box_reg" : args.loss_weights[1]}
loss_fn = rpn_loss_fn(weights)

collate_fn = rpn_collate_fn(use_built_in_rpn)

run_folder = 'region_proposal_network'

network_trainer = NetworkTrainer(
    data       = rpn_dataset,
    model      = rpn,
    optimizer  = optimizer,
    run_folder = run_folder,
    num_epochs = 20,
    batch_size = 128,
    log_fn     = rpn_log_fn,
    loss_fn    = loss_fn,
    collate_fn = collate_fn)

network_trainer.train()
