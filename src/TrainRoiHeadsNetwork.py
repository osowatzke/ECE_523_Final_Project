from BackboneNetwork        import BackboneNetwork
from DataManager            import DataManager
from FlirDataset            import FlirDataset
from NetworkTrainer         import NetworkTrainer
from PathConstants          import PathConstants
from RegionProposalNetwork  import *
from networkHead            import *

import argparse
import os

# Parse optional input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--loss_weights', nargs=2, default=[1,1], type=float)
parser.add_argument('-l', '--learning_rate', default=1e-3, type=float)
args = parser.parse_args()

# Determine the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# Free GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Flags to switch between built-in and custom implementations
use_built_in_rpn = False
use_built_in_roi_heads = True

# Create Data Manager Object
# Will download dataset if not already available
data_manager = DataManager()
data_manager.download_datasets()

# Create Path Constants Singleton
data_dir = data_manager.get_download_dir()
PathConstants(data_dir)

# Create input dataset
dataset = FlirDataset(PathConstants.TRAIN_DIR, num_images=10, device=device)

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

# Load User Weights
curr_file_path = os.path.dirname(__file__)
weights_path = os.path.join(curr_file_path,'weights','rpn_weights.pth')
state_dict = torch.load(weights_path, map_location=device)
rpn.load_state_dict(state_dict['model_state'])

# Create ROI Heads dataset
rpn.eval()
roi_dataset = create_roi_dataset(rpn, dataset, rpn_dataset, use_built_in_roi_heads, device)

# Clear unneeded things from GPU memory
del dataset
del backbone
del rpn
del rpn_dataset

# Free GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Create ROI Heads network
torch.manual_seed(0)
roi_heads = create_roi_heads_network(feature_map_size, use_built_in_roi_heads)
roi_heads.to(device)
roi_heads.train()

# Train the RPN network
optimizer = torch.optim.SGD(
    roi_heads.parameters(),
    lr = args.learning_rate,
    momentum = 0.9,
    weight_decay = 5e-3)

collate_fn = roi_collate_fn(use_built_in_roi_heads)

# Create loss function with user weights
weights = {
    "loss_classifier"  : args.loss_weights[0],
    "loss_box_reg"     : args.loss_weights[1]}
loss_fn = roi_loss_fn(weights, use_built_in_roi_heads)

# Train ROI heads network
run_folder = 'roi_heads_network'

network_trainer = NetworkTrainer(
    data       = roi_dataset,
    model      = roi_heads,
    optimizer  = optimizer,
    run_folder = run_folder,
    num_epochs = 50,
    batch_size = 1,
    log_fn     = roi_log_fn,
    loss_fn    = loss_fn,
    collate_fn = collate_fn)

network_trainer.train()
