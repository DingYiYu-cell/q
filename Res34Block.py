import torch.nn as nn
from torchvision import models
from CBAM import CBAM
class Res34Block(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        # 1. 搬出大神练好的 ResNet34 喵！
        resnet = models.resnet34(pretrained=False)
        
        # 2. 把零件按层级拆开，方便做“跳跃连接”喵
        self.first_layer = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.pool = resnet.maxpool
        self.layer1 = resnet.layer1  # 对应第一层特征
        self.layer2 = resnet.layer2  # 对应第二层特征
        self.layer3 = resnet.layer3  # 对应第三层特征
        self.layer4 = resnet.layer4  # 底部 Bottleneck 喵！
        self.cbam64 = CBAM(64)
        self.cbam128 = CBAM(128)
        self.cbam256 = CBAM(256)
        self.cbam512 = CBAM(512)

    def forward(self, x):
        # 每一层算完都要留个“备份”，传给右边的 Decoder 喵
        
        x0 = self.first_layer(x) # 64通道
        x1 = self.layer1(self.pool(x0)) # 64通道
        x1 = self.cbam64(x1)
        x2 = self.layer2(x1) # 128通道
        x2 = self.cbam128(x2)
        x3 = self.layer3(x2) # 256通道
        x3 = self.cbam256(x3)
        x4 = self.layer4(x3) # 512通道 (最底层)\
        x4 = self.cbam512(x4)
        
        return [x0, x1, x2, x3, x4]