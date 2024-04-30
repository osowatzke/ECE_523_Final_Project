import torch
x = [torch.rand(3,512,640),torch.rand(3,512,640)]
y = torch.stack((*x,))
print(y.shape)