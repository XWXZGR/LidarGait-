import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from torchvision.models.resnet import ResNet, BasicBlock
from ..modules import BasicConv2d

# Define IBN module
class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

# Define SE module
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Modify BasicBlock to include IBN and SE
class Block(BasicBlock):
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_ibn=False, use_se=False):
        super(Block, self).__init__(inplanes, planes, stride, downsample)
        norm_layer = IBN if use_ibn else nn.BatchNorm2d
        self.bn1 = norm_layer(self.conv1.out_channels)
        self.bn2 = norm_layer(self.conv2.out_channels)
        self.se = SELayer(self.conv2.out_channels) if use_se else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.se:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class resnet_se_ibn(ResNet):
    def __init__(self, block=Block, layers=[1, 2, 2, 1], channels=[32, 64, 128, 256], in_channel=1, strides=[1, 2, 2, 1], maxpool=True, use_ibn=False, use_se=False):

        self.use_ibn = use_ibn
        self.use_se = use_se
        self.maxpool_flag = maxpool

        # Initialize ResNet with the custom block
        super(resnet_se_ibn, self).__init__(block, layers)
        # Not used #
        self.fc = None
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        # Replace layers with custom ones
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0], dilate=False, use_ibn=use_ibn, use_se=use_se)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1], dilate=False, use_ibn=use_ibn, use_se=use_se)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2], dilate=False, use_ibn=use_ibn, use_se=use_se)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3], dilate=False, use_ibn=use_ibn, use_se=use_se)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, use_ibn=False, use_se=False):
        norm_layer = IBN if use_ibn else nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_ibn=use_ibn, use_se=use_se))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_ibn=use_ibn, use_se=use_se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_flag:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
