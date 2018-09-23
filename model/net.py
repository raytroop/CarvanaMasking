import numpy as np
from torchvision import models
import torch
from torch import nn
from .resnet_conv import resnet34_bottom


class naive_net(nn.Module):
    def __init__(self, fresh=False):
        super().__init__()
        self.feats = resnet34_bottom(pretrained=False)
        if fresh:
            resnet34_dict = torch.load('/home/raytroop/.torch/models/resnet34-333f7ec4.pth')
            resnet34_dict.pop('fc.weight')
            resnet34_dict.pop('fc.bias')
            self.feats.load_state_dict(resnet34_dict)

        self.simple_up = nn.Sequential(
            nn.ReLU(),
            self._make_StdUpsample_layer(512, 256),
            self._make_StdUpsample_layer(256, 256),
            self._make_StdUpsample_layer(256, 256),
            self._make_StdUpsample_layer(256, 256),
            nn.ConvTranspose2d(256, 1, 2, stride=2))


    def _make_StdUpsample_layer(self, nin, nout):
        return nn.Sequential(nn.ConvTranspose2d(nin, nout, 2, stride=2),
                             nn.ReLU(),
                             nn.BatchNorm2d(nout)
                             )
    def forward(self, x):
        x = self.feats(x)
        x = self.simple_up(x)
        return x

    def freeze_bottom(self):
        for p in self.feats.parameters():
            p.requires_grad = False
        self.feats.eval()

def accuracy(preds, target):
    """
    compute accuracy per pixel

    preds and target share same shape,they are np.ndarray
    """
    return np.mean((preds > 0.0) == target)

# maintain all metrics required in this dictionary-
# these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
