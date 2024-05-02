import torch
import math

# Function converts bounding boxes from corners representation
# to centroid representation:
def corners_to_centroid(bboxes, anchor_boxes):

    # print(bboxes.shape)
    # print(anchor_boxes.shape)

    # Extract the four corners of the bounding box
    xmin = bboxes[:,0]
    ymin = bboxes[:,1]
    xmax = bboxes[:,2]
    ymax = bboxes[:,3]

    # Convert to width and height
    w = xmax - xmin
    h = ymax - ymin

    # Get box centers
    x = xmin + w/2
    y = ymin + h/2

    # Extract the four corners of the anchor boxes
    xmin = anchor_boxes[:,0]
    ymin = anchor_boxes[:,1]
    xmax = anchor_boxes[:,2]
    ymax = anchor_boxes[:,3]

    # Convert to width and height
    wa = xmax - xmin
    ha = ymax - ymin

    # Get box centers
    xa = xmin + wa/2
    ya = ymin + ha/2

    # Convert to centroid representation
    tx = torch.divide(x - xa, wa)
    ty = torch.divide(y - ya, ha)
    tw = torch.log(torch.divide(w, wa))
    th = torch.log(torch.divide(h, ha))

    # Pack as a Nx4 matrix
    return torch.stack((tx,ty,tw,th),axis=1)

# Function converts bounding boxes from centroids representation
# to corners representation:
def centroids_to_corners(offsets, anchor_boxes):

    # Repeat anchor boxes to match size of offsets
    num_batches = offsets.shape[0]//anchor_boxes.shape[0]
    anchor_boxes = anchor_boxes.repeat(num_batches, 1)

    #print(offsets)
    #print(anchor_boxes)

    # Extract the four corners of the bounding box
    tx = offsets[:,0]
    ty = offsets[:,1]
    tw = offsets[:,2]
    th = offsets[:,3]

    # Extract the four corners of the anchor boxes
    # print(anchor_boxes.shape)
    xmin = anchor_boxes[:,0]
    ymin = anchor_boxes[:,1]
    xmax = anchor_boxes[:,2]
    ymax = anchor_boxes[:,3]

    # Convert to width and height
    wa = xmax - xmin
    ha = ymax - ymin

    print(wa)
    print(ha)

    # Get box centers
    xa = xmin + wa/2
    ya = ymin + ha/2

    print(xa)
    print(ya)
    
    # Determine centers
    x = tx*wa + xa
    y = ty*ha + ya
    w = wa*torch.exp(torch.clamp(tw, max=math.log(1000.0/16)))
    h = ha*torch.exp(torch.clamp(th, max=math.log(1000.0/16)))

    # Get corners representation
    xmin = x - w/2
    ymin = y - h/2
    xmax = xmin + w
    ymax = ymin + h

    # Pack as a Nx4 matrix
    return torch.stack((xmin,ymin,xmax,ymax),axis=1)

