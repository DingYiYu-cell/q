import torch.nn as nn
# class ResBlock(nn.Module):#残差卷积块 类
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
#         self.mp = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels)
#         )
#         self.shortcut = nn.Sequential()#恒等映射
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         out = self.mp(x)
#         out += self.shortcut(x)
#         return self.relu(out)#两条线路会和后再ReLU

import torch
import torch.nn as nn
from torchvision import models

class ResNet34UnetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # 1. 搬出大神练好的 ResNet34 喵！
        resnet = models.resnet34(pretrained=pretrained)
        
        # 2. 把零件按层级拆开，方便做“跳跃连接”喵
        self.first_layer = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1  # 对应第一层特征
        self.layer2 = resnet.layer2  # 对应第二层特征
        self.layer3 = resnet.layer3  # 对应第三层特征
        self.layer4 = resnet.layer4  # 底部 Bottleneck 喵！

    def forward(self, x):
        # 每一层算完都要留个“备份”，传给右边的 Decoder 喵
        x0 = self.first_layer(x) # 64通道
        x1 = self.layer1(self.pool(x0)) # 64通道
        x2 = self.layer2(x1) # 128通道
        x3 = self.layer3(x2) # 256通道
        x4 = self.layer4(x3) # 512通道 (最底层)
        return [x0, x1, x2, x3, x4]