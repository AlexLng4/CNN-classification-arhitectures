import torch
from torch import nn
import numpy as np

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()

        # expected input is 227x227x3
        # after the first 2 conv layers we will apply an old method for normalization of the big data, LRN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        # -> 55x55x96
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # -> 27x27x96
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # -> 27x27x256
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # -> 13x13x256
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # -> 13x13x384
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # -> 13x13x384
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # -> 13x13x256
        self.pooling5= nn.MaxPool2d(kernel_size=3, stride=2)
        # -> 6x6x256

        # flattening to 9216
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=9216, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # apply then LRN
        x = self.pooling1(x)
        x = torch.relu(self.conv2(x))
        # apply then LRN
        x = self.pooling2(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = self.pooling5(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

if __name__=="__main__":

    net = AlexNet()
    print(net)
    # test element with batch
    test_array = torch.randn(1, 3, 227, 227)
    out = net(test_array)
    print(out.shape)