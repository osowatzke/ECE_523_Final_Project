import numpy as np
import torch
from torchvision import ops

class AnchorBoxGenerator:
    def __init__(self, image_size, feature_map_size, scales=[2, 4, 6], ratios=[0.5, 1, 1.5]):

        # Define member variables
        self.anchor_boxes = None
        self.anchor_boxes_proj = None
        
        # Generate anchor boxes using feature map
        self.generate_anchor_boxes(feature_map_size, scales=[2, 4, 6], ratios=[0.5, 1, 1.5])
        
        # Project anchor boxes onto the original image
        self.project_anchor_boxes(image_size, feature_map_size)

    def generate_anchor_boxes(self, size, scales, ratios):

        # Extract height and width from size tuple
        height = size[0]
        width = size[1]

        # Convert scales and ratios to 4D numpy arrays
        scales = np.array(scales)
        ratios = np.array(ratios)
        scales = np.reshape(scales,(1,1,-1,1))
        ratios = np.reshape(ratios,(1,1,1,-1))

        # Convert anchor box centers to 4D numpy arrays
        anchor_box_center_x = np.arange(width) + 0.5
        anchor_box_center_y = np.arange(height) + 0.5
        anchor_box_center_x = np.reshape(anchor_box_center_x, (1,-1,1,1))
        anchor_box_center_y = np.reshape(anchor_box_center_y, (-1,1,1,1))

        # Compute the height and width of each anchor box
        h = scales*np.ones(ratios.shape)
        w = scales*ratios

        # Determine the corners of the anchor boxes
        ymin = anchor_box_center_y - h/2
        ymax = anchor_box_center_y + h/2
        xmin = anchor_box_center_x - w/2
        xmax = anchor_box_center_x + w/2

        # Convert outputs into a vector
        ymin = ymin.ravel()
        ymax = ymax.ravel()
        xmin = xmin.ravel()
        xmax = xmax.ravel()

        # Define an Nx4 matrix of anchor boxes
        self.anchor_boxes = np.transpose(np.array([xmin, ymin, xmax, ymax]))
        self.anchor_boxes = torch.Tensor(self.anchor_boxes)

        # Ensure anchor boxes are contained within bounds of image
        self.anchor_boxes = ops.clip_boxes_to_image(self.anchor_boxes, (height, width))
        
    def project_anchor_boxes(self, image_size, feature_map_size):

        # Determine scale factors to go from feature map coordinates to image coordinates
        height_sf = image_size[0]/feature_map_size[0]
        width_sf = image_size[1]/feature_map_size[1]

        # Initiliaze projected anchor boxes with anchor boxes
        self.anchor_boxes_proj = self.anchor_boxes.detach().clone()

        # Project each coordinate of the anchor boxes back onto the image
        self.anchor_boxes_proj[:,0::2] = self.anchor_boxes_proj[:,0::2] * width_sf
        self.anchor_boxes_proj[:,1::2] = self.anchor_boxes_proj[:,1::2] * height_sf

if __name__ == "__main__":
    anchor_box_generator = AnchorBoxGenerator(image_size=(32,32),feature_map_size=(8,8))
    print(anchor_box_generator.anchor_boxes)
    print(anchor_box_generator.anchor_boxes_proj)
