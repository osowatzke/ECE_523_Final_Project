import torch.nn as nn
import torch.nn.functional as F

class RegionProposalHead(nn.Module):
    def __init__(self,in_channels,num_anchors,hidden_layer_size=512):
        super.__init__()
        self.conv1 = nn.Conv2d(in_channels,hidden_layer_size,3,padding=1)
        self.conv2 = nn.Conv2d(hidden_layer_size,num_anchors,1)
        self.conv3 = nn.Conv2d(hidden_layer_size,4*num_anchors,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        scores = self.conv2(x)
        offsets = self.conv3(x)

    
        pred = {}
        pred['scores'] = self.conv2(x)
        pred['offsets'] = self.conv3(x)
        loss = None
        if self.training:
            loss = {}
            loss['scores'] = F.binary_cross_entropy_with_logits(pred['scores'],gt['scores'])
            loss['offsets'] = F.smooth_l1_loss(pred['offsets'],gt['offsets'])
        return pred, loss
    

class RegionProposalNetwork(nn.Module)
    def __init__(self, anchor_box_generator):
        