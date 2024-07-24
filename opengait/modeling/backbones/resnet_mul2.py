import torch
from torch.nn import functional as F
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ..modules import BasicConv2d

block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet9_mul2(ResNet):
    def __init__(self, block, channels=[64, 128, 256, 512], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                 maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(ResNet9_mul2, self).__init__(block, layers)

        # Not used #
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=strides[0], dilate=False)

        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], dilate=False)

        #输入张量的大小从 [640, 64, 64, 64] 转换为 [640, 512, 16, 16]
        self.pyramid_conv1 = nn.Conv2d(64, 512, kernel_size=4, stride=4, bias=False).cuda()
        #self.pyramid_conv2 = nn.Conv2d(channels[1], 512, kernel_size=1, stride=1, bias=False).cuda()
        self.pyramid_conv2 = nn.Conv2d(channels[1], 512, kernel_size=3, stride=2, padding=1, bias=False).cuda()
        #self.pyramid_conv3 = nn.Conv2d(channels[2], 512, kernel_size=1, stride=1, bias=False).cuda()
        self.pyramid_conv4 = nn.Conv2d(channels[3], 512, kernel_size=1, stride=1, bias=False).cuda()

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.maxpool_flag:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        print('x:',x.shape)
        p1 = self.pyramid_conv1(x)

        c1 = self.layer1(x)
        print("c1:",c1.shape)
        c2 = self.layer2(c1)
        print("c2:", c2.shape)
        c3 = self.layer3(c2)
        print("c3:", c3.shape)
        c4 = self.layer4(c3)
        print("c4:", c4.shape)

        # 构建特征金字塔
        p4 = self.pyramid_conv4(c4)
        print("p4:", p4.shape)

        p2 = self.pyramid_conv2(c2)
        print("p2:", p2.shape)


        # 融合所有尺度的特征
        fused_feature = p4 + p1 + p2
        print('fused_feature:', fused_feature.shape)
        c4 = fused_feature

        return c4