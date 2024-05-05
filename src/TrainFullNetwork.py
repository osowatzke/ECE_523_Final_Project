from BackboneNetwork        import BackboneNetwork
from DataManager            import DataManager
from FlirDataset            import FlirDataset
from NetworkTrainer         import NetworkTrainer
from PathConstants          import PathConstants
from fasterRCNNloss         import fasterRCNNloss
from RegionProposalNetwork  import *
from networkHead            import *
from fasterRCNN             import *

import argparse
import os

# Flags to switch between built-in and custom implementations
use_built_in_rpn = False
use_built_in_roi_heads = True

# Parse optional input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-w', '--loss_weights', nargs=4, default=[1,1,1,1], type=float)
parser.add_argument('-l', '--learning_rate', default=1e-4, type=float)
parser.add_argument('-n', '--num_images', default=-1, type=int)
parser.add_argument('-b', '--batch_size', default=96, type=int)
args = parser.parse_args()

# Determine the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

# Free GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Create Data Manager Object
# Will download dataset if not already available
data_manager = DataManager()
data_manager.download_datasets()

# Create Path Constants Singleton
data_dir = data_manager.get_download_dir()
PathConstants(data_dir)

# Create input dataset
train_data = FlirDataset(PathConstants.TRAIN_DIR, num_images=args.num_images, device=device)

# Create Faster RCNN network with pretrained layers
image_size = train_data[0][0].shape
feature_map_size = (1, 2048, 16, 20)

# Create RPN
torch.manual_seed(0)
rpn = create_region_proposal_network(image_size, feature_map_size, use_built_in_rpn)
rpn.to(device)
rpn.train()

# Load User Weights
curr_file_path = os.path.dirname(__file__)
weights_path = os.path.join(curr_file_path,'weights','rpn_weights.pth')
state_dict = torch.load(weights_path, map_location=device)
rpn.load_state_dict(state_dict['model_state'])

# Create ROI Heads network
torch.manual_seed(0)
roi_heads = create_roi_heads_network(feature_map_size, use_built_in_roi_heads)
roi_heads.to(device)
roi_heads.train()

# Load User Weights
curr_file_path = os.path.dirname(__file__)
weights_path = os.path.join(curr_file_path,'weights','roi_weights.pth')
state_dict = torch.load(weights_path, map_location=device)
roi_heads.load_state_dict(state_dict['model_state'])

model = FasterRCNN(
    image_size              = image_size,
    region_proposal_network = rpn,
    roi_heads_network       = roi_heads,
    use_built_in_rpn        = use_built_in_rpn,
    use_built_in_roi_heads  = use_built_in_roi_heads)

model.to(device)

# Create optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

# Set the period for saving data
# -1 will cause data not to be saved
save_period = {'epoch' : 1, 'batch' : -1}

# Loss function
rcnn_loss_fn = fasterRCNNloss({
    "loss_objectness"  : args.loss_weights[0],
    "loss_rpn_box_reg" : args.loss_weights[1],
    "loss_classifier"  : args.loss_weights[2],
    "loss_box_reg"     : args.loss_weights[3]})

# Run subfolder
run_folder = 'four_stage_faster_rcnn'

# Create network trainer
net_trainer = NetworkTrainer(
    data        = train_data, 
    model       = model,
    optimizer   = optimizer,
    run_folder  = run_folder,
    num_epochs  = 50,
    batch_size  = args.batch_size,
    loss_fn     = rcnn_loss_fn,
    log_fn      = rcnn_log_fn,
    collate_fn  = rcnn_collate_fn,
    save_period = save_period,
    device      = device
)

net_trainer.train()