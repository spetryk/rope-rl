import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.conv_layers = nn.Sequential(
            # input is batch_size //x 3 x 16 x 32
            nn.Conv2d(3, 24, 3, stride=2, bias=False),
            nn.ELU(),
            nn.Conv2d(24, 48, 3, stride=2, bias=False),
            nn.MaxPool2d(4, stride=4),
            nn.Dropout(p=0.25)
        )
        self.linear_layers = nn.Sequential(
            #input from sequential conv layers
            nn.Linear(in_features=48*4*768, out_features=50, bias=False),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=20, bias=False),
            nn.Linear(in_features=20, out_features=7, bias=False),
        )
        self._initialize_weights()
        
    # custom weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                init.normal(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                init.normal(m.weight, mean=1, std=0.02)
                init.constant(m.bias, 0)

    def forward(self, input):
        input = input.view(input.size(0), 772, 1032, 3)
        input = input.permute(0, 3, 1, 2)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output    

class ResNet18(nn.Module):
    def __init__(self, pretrained=False, channels=3):
        super(ResNet18, self).__init__()
        m = torchvision.models.resnet18(pretrained=pretrained)
        m.fc = nn.Linear(in_features=512, out_features=7)
        self.channels = channels

        if channels == 1:
            assert not pretrained, "Pretrained model can only be used with 3-channel image"
            m.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        self.model = m

    def forward(self, x):
        return self.model(x)



