import torch


img = all_imgs[0][0]
torch.from_numpy(img).type(torch.float32)