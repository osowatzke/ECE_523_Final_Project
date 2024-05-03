import torch

def sample_data(all_labels, num_anchors, batch_size, pos_frac):

    # Create empty lists for positive and negative samples
    pos_samp = []
    neg_samp = []

    # Offset into flattened vector
    offset = 0

    # Loop over the labels from each image
    for labels in torch.split(all_labels, num_anchors):

        # Compute positive and negative indices
        pos_idx = torch.where(labels == 1)[0]
        neg_idx = torch.where(labels == 0)[0]
        
        # Determine the maximum number of positive samples
        max_pos = batch_size*pos_frac

        # Determine the actual number of positive samples
        num_pos = min(pos_idx.numel(), max_pos)

        # Determine the number of negative samples
        num_neg = batch_size - num_pos

        # Select a random subset of the positive samples
        rand_idx = torch.randperm(pos_idx.numel())[:num_pos]
        pos_idx = pos_idx[rand_idx]
        
        # Select a random subset of the negative samples
        rand_idx = torch.randperm(neg_idx.numel())[:num_neg]
        neg_idx = neg_idx[rand_idx]

        # Account for index offsetes
        pos_idx += offset
        neg_idx += offset

        # Update offset
        offset += num_anchors

        # Append positive and negative samples to list
        pos_samp.append(pos_idx)
        neg_samp.append(neg_idx)

    # Create tensors using list data
    pos_samp = torch.cat(pos_samp)
    neg_samp = torch.cat(neg_samp)
    
    return pos_samp, neg_samp