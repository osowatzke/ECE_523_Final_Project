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
        anchor_box_center_x = np.repeat(anchor_box_center_x, anchor_box_center_y.shape[0], axis=0)
        anchor_box_center_y = np.repeat(anchor_box_center_y, anchor_box_center_x.shape[1], axis=1)
        
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

    def get_training_data(self, bounding_boxes_truth):

        # Determine the IOU between the anchor boxes and true bounding boxes
        iou = ops.box_iou(self.anchor_boxes_proj, bounding_boxes_truth)

        # Determine which anchor boxes correspond to the maximum IOU
        max_idx = torch.max(iou, dim=0)[1]
        max_label = torch.zeros(iou.shape[0], dtype=torch.bool)
        max_label[max_idx] = 1

        # Determine which samples have true and false labels
        true_label = torch.sum(iou > 0.7, dim=1) > 0
        true_label = true_label | max_label
        false_label = torch.sum(iou < 0.3, dim=1) == iou.shape[1]

        # Get indices of true and false labels
        idx = np.arange(iou.shape[0])
        true_idx = idx[true_label]
        false_idx = idx[false_label]

        # Shuffle true and false indices
        true_idx = true_idx[np.random.permutation(len(true_idx))]
        false_idx = false_idx[np.random.permutation(len(false_idx))]

        # Determine how many true and false samples to take
        # Attempt to take 128 of each class
        num_true_samples = min(len(true_idx),128)
        num_false_samples = 256 - num_true_samples

        # Select subset of true and false indices
        true_idx = true_idx[0:num_true_samples]
        false_idx = false_idx[0:num_false_samples]

        # Determine which samples to select
        selected_sample = torch.zeros(iou.shape[0], dtype=torch.bool)
        selected_sample[true_idx] = 1
        selected_sample[false_idx] = 1

        # Select which anchor have labels
        anchor_boxes_training = self.anchor_boxes.detach().clone()
        anchor_boxes_training = anchor_boxes_training[selected_sample]

        # Create array of labels for the training data
        labels = torch.zeros(iou.shape[0])
        labels[true_label] = 1
        labels = labels[selected_sample]

        # Return anchor boxes and labels for training
        return(anchor_boxes_training, labels)

if __name__ == "__main__":
    anchor_box_generator = AnchorBoxGenerator(image_size=(600,600),feature_map_size=(16,20))
    (anchor_boxes, labels) = anchor_box_generator.get_training_data(torch.Tensor([[0,0,4,4]]))
    print(anchor_boxes.shape)
    print(labels.shape)