
import torch
import torch.nn.functional as F
from torchvision import transforms

def convert_2_tensor(img):
    return transforms.functional.to_tensor(img).unsqueeze(0)
