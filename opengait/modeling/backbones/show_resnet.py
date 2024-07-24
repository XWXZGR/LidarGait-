import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from opengait.modeling.modules import BasicConv2d
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

block_map = {'BasicBlock': BasicBlock, 'Bottleneck': Bottleneck}

class resnet_mul3_test(ResNet):
    def __init__(self, block, channels=[64, 128, 256, 512], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                 maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError("Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(resnet_mul3_test, self).__init__(block, layers)

        # Not used #
        self.fc = None
        ############
        self.inplanes = channels[0]
        self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.conv1 = BasicConv2d(in_channel, self.inplanes, 3, 1, 1)

        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=strides[0], dilate=False)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=strides[1], dilate=False)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=strides[2], dilate=False)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=strides[3], dilate=False)

        self.pyramid_conv1 = nn.Conv2d(channels[0], 512, kernel_size=1, stride=1, bias=False)
        self.pyramid_conv2 = nn.Conv2d(channels[1], 512, kernel_size=1, stride=1, bias=False)
        self.pyramid_conv3 = nn.Conv2d(channels[2], 512, kernel_size=1, stride=1, bias=False)
        self.pyramid_conv4 = nn.Conv2d(channels[3], 512, kernel_size=1, stride=1, bias=False)

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

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # 构建特征金字塔
        p4 = self.pyramid_conv4(c4)
        p3 = self.pyramid_conv3(c3)
        p2 = self.pyramid_conv2(c2)
        p1 = self.pyramid_conv1(c1)

        # 降采样
        p2 = F.interpolate(p2, scale_factor=0.5, mode='nearest')
        p1 = F.interpolate(p1, scale_factor=0.5, mode='nearest')
        p1 = F.interpolate(p1, scale_factor=0.5, mode='nearest')

        # 融合所有尺度的特征
        fused_feature = p1 + p2 + p3 + p4
        print("p1",p1.shape)

        return p1[:, -1].unsqueeze(0).unsqueeze(0),p2[:, -1].unsqueeze(0).unsqueeze(0),p3[:, -1].unsqueeze(0).unsqueeze(0),p4[:, -1].unsqueeze(0).unsqueeze(0),fused_feature[:, -1].unsqueeze(0).unsqueeze(0)

# 加载图像并进行预处理
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0)
    return image

# 图像路径
image_path = r"E:\Dataset\SUSTech1K\SUSTech1K-Released-2023\0001\00-nm\315\PCDs_depths\1657079573.500289000.png"

# 创建模型实例
model = resnet_mul3_test(block='BasicBlock')

# 加载并预处理图像
input_image = load_image(image_path)

# 前向传播
model.eval()
with torch.no_grad():
    p1, p2, p3, p4,fused_feature = model(input_image)

# # 可视化特征图
# #plt.imshow(p1.squeeze().cpu().detach().numpy(), cmap='viridis')
# plt.imshow(p1.squeeze().cpu().detach().numpy(), cmap='viridis', extent=(0, p1.shape[2], 0, p1.shape[2]))
# plt.axis('off')
# plt.savefig('p1_feature_map.png', bbox_inches='tight', pad_inches=0)
# plt.show()
#
# plt.imshow(p2.squeeze().cpu().detach().numpy(), cmap='viridis', extent=(0, p2.shape[2], 0, p2.shape[2]))
# plt.axis('off')
# plt.savefig('p2_feature_map.png', bbox_inches='tight', pad_inches=0)
# plt.show()
#
# plt.imshow(p3.squeeze().cpu().detach().numpy(), cmap='viridis', extent=(0, p3.shape[2], 0, p3.shape[2]))
# plt.axis('off')
# plt.savefig('p3_feature_map.png', bbox_inches='tight', pad_inches=0)
# plt.show()
#
# plt.imshow(p4.squeeze().cpu().detach().numpy(), cmap='viridis', extent=(0, p4.shape[2], 0, p4.shape[2]))
# plt.axis('off')
# plt.savefig('p4_feature_map.png', bbox_inches='tight', pad_inches=0)
# plt.show()
#
# plt.imshow(fused_feature.squeeze().cpu().detach().numpy(), cmap='viridis', extent=(0, fused_feature.shape[2], 0, fused_feature.shape[2]))
# plt.axis('off')
# plt.savefig('fused_feature_map.png', bbox_inches='tight', pad_inches=0)
# plt.show()