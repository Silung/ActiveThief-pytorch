import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import matplotlib.pyplot as plt

class ResNet_Pretrained(nn.Module):
    def __init__(self, num_classes, type='18'):
        try:
            type_int = int(type)
        except:
            type = '18'            
        super(ResNet_Pretrained, self).__init__()
        if type == '18':
            self.model = resnet18(weights=None)
            weights = ResNet18_Weights.IMAGENET1K_V1
            in_features = 512
        elif type == '34':
            self.model = resnet34(weights=None)
            weights = ResNet34_Weights.IMAGENET1K_V1
            in_features = 1024
        elif type == '50':
            self.model = resnet50(weights=None)
            weights = ResNet50_Weights.IMAGENET1K_V1
            in_features = 2048
        self.preprocess = weights.transforms()
        self.model = resnet18(weights=weights)
        self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    
    def forward(self, x):
        # Input B H W C
        x = x.float().permute(0,3,1,2)
        # if self.training:
        x = self.preprocess(x)
        x = self.model(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, num_classes, type='18'):
        super(ResNet, self).__init__()
        print(f'Model type: resnet{type}')
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.preprocess = weights.transforms()
        if type == '18':
            self.model = resnet18(weights=None)
            in_features = 512
        elif type == '34':
            self.model = resnet34(weights=None)
            in_features = 1024
        elif type == '50':
            self.model = resnet50(weights=None)
            in_features = 2048
        self.model.fc = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    
    def forward(self, x):
        # Input B H W C
        x = x.float().permute(0,3,1,2)
        # if self.training:
        x = self.preprocess(x)
        x = self.model(x)
        return x


