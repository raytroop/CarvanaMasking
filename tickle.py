"""
    this is what
"""
from torchvision import models
import torch
from torch import nn, optim
import model.resnet_conv as resnet_conv
from model.net import naive_net

model = naive_net()
optimizer = optim.Adam(model.parameters(), lr=0.001) # pylint: disable=E1101
print(optimizer.state_dict())
print('///////////////////////////////')
optimizer = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
print(optimizer.state_dict())
optimizer.step()
print(optimizer.state_dict())
# print(optimizer.param_groups)