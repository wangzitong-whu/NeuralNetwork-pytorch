import torch
from torch import nn


class LeNET(nn.Module):
    def __init__(self):
        super(LeNET, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5,1,2)  # input输入通道数，output输出通道数，kernel卷积核大小
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2) #14
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
