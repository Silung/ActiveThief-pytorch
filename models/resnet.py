import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet_Pretrained(nn.Module):
    def __init__(self):
        super(ResNet_Pretrained, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.preprocess = weights.transforms()
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    
    def forward(self, x):
        # Input B H W C
        x = x.float().permute(0,3,1,2)
        x = self.preprocess(x)
        x = self.model(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.preprocess = weights.transforms()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    
    def forward(self, x):
        # Input B H W C
        x = x.float().permute(0,3,1,2)
        x = self.preprocess(x)
        
        x = self.model(x)
        return x


