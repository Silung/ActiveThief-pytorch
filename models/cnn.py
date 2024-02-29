import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, convs_in_block, drop_prob=0.1):
        super(ConvBlock, self).__init__()

        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm2d(out_channels, momentum=0.01))

        for _ in range(convs_in_block - 1):  # 2 repeated units
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.01))
        
        self.conv_blocks = nn.Sequential(*layers)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=drop_prob)
    
    def forward(self, x):
        x = self.conv_blocks(x)
        
        x = self.pool(x)
        x = self.dropout(x)
        
        return x

class CNN(nn.Module):
    def __init__(self, num_classes, dataset_name, in_channels=None, num_filters=[32, 64, 128], fc_layers=[], convs_in_block=2, drop_prob=0.1):
        super(CNN, self).__init__()

        self.num_filters      = num_filters
        self.fc_layers        = fc_layers 
        self.convs_in_block   = convs_in_block
        self.drop_prob = drop_prob
        
        layers = []
                
        if 'mnist' in dataset_name:
            img_size = 28
        elif 'cifar' in dataset_name:
            img_size = 32
        elif 'imagenet' in dataset_name:
	        img_size = 64
        elif 'gtsrb' in dataset_name:
	        img_size = 32
        else:
            raise NotImplementedError
        
        ex_out_channels = in_channels
        for out_channels in num_filters:
            layers.append(ConvBlock(ex_out_channels, out_channels, convs_in_block, drop_prob=drop_prob))
            ex_out_channels = out_channels
            img_size = img_size // 2

        
        ex_out_channels *= img_size ** 2
        for num_neurons in self.fc_layers:
            layers.append(nn.Linear(ex_out_channels, num_neurons))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.drop_prob))
            ex_out_channels = num_neurons
        self.conv_blocks = nn.Sequential(*layers)

        self.fc = nn.Linear(ex_out_channels, num_classes)
    
    def forward(self, x):
        # input shape H*W*C
        x = x.float().permute(0,3,1,2)
        x = self.conv_blocks(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

