import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model.SPP import ASPP_simple, ASPP
from model.ResNet import ResNet101, ResNet18, ResNet34, ResNet50

import time


class ResNet_ASPP(nn.Module):
    def __init__(self, nInputChannels, os, backbone_type, pretrained):
        super(ResNet_ASPP, self).__init__()

        self.os = os
        self.backbone_type = backbone_type

        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        elif os == 32:
            rates = [1, 6, 12, 18]
        else:
            raise NotImplementedError

        if backbone_type == 'resnet18':
            self.backbone_features = ResNet18(nInputChannels, os, pretrained=False)
        elif backbone_type == 'resnet34':
            self.backbone_features = ResNet34(nInputChannels, os, pretrained=True)
        elif backbone_type == 'resnet50':
            self.backbone_features = ResNet50(nInputChannels, os, pretrained=pretrained)
        else:
            raise NotImplementedError
        if backbone_type == 'resnet50':
            asppInputChannels = 512*4
            asppOutputChannels = 128

        self.aspp = ASPP(asppInputChannels, asppOutputChannels, rates)


    def forward(self, input):
        x, low_level_features, conv1_feat, layer2_feat, layer3_feat = self.backbone_features(input)
        layer4_feat = x
        x = self.aspp(x)

        return conv1_feat, low_level_features, layer2_feat, layer3_feat, layer4_feat, x