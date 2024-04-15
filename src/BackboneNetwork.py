from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

# Pretrained reset50 Backbone Network
class BackboneNetwork(nn.Sequential):
    def __init__(self):
        super().__init__()

        # Load Pretrained resnet50 Network
        backbone_network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Remove Last Two Layers of Backbone Network
        backbone_layers = list(backbone_network.children())[:-2]

        # Add backbone layers to module
        for layer in backbone_layers:
            self.append(layer)

        # Fix backbone network weights
        for param in self.parameters():
            param.requires_grad = False
            
if __name__ == "__main__":

    # Create backbone network
    backbone_network = BackboneNetwork()

    # List layers of backbone network
    print(list(backbone_network.children()))


