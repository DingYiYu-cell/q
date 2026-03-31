import torch.nn as nn
from torchvision import models
from CBAM import CBAM
class Res34Block(nn.Module):
    def __init__(self, weights=None):
        super().__init__()

        resnet34 = models.resnet34(weights=None)
        resnet34.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)

        self.stage0 = nn.Sequential(resnet34.conv1, resnet34.bn1, resnet34.relu) 
        self.pool = resnet34.maxpool
        #下面分开写是因为每一层都要做备份
        self.stage1 = resnet34.layer1  
        self.stage2 = resnet34.layer2  
        self.stage3 = resnet34.layer3  
        self.stage4 = resnet34.layer4  # 底部 Bottleneck 
        
        #CBAM 空间通道混合注意力
        self.cbam64 = CBAM(64)
        self.cbam128 = CBAM(128)
        self.cbam256 = CBAM(256)
        self.cbam512 = CBAM(512)
        
        self.drop = nn.Dropout2d(p=0.2)#!随机失活
    def forward(self, x):
        
        x0 = self.stage0(x) # 64通道
        x1 = self.stage1(self.pool(x0)) # 64通道
        x1 = self.cbam64(x1)
        x2 = self.stage2(x1) # 128通道
        x2 = self.cbam128(x2)
        x3 = self.stage3(x2) # 256通道
        x3 = self.cbam256(x3)
        x4 = self.stage4(x3) # 512通道 (最底层)\
        x4 = self.cbam512(x4)
        x4 = self.drop(x4)
        
        return [x0, x1, x2, x3, x4]#做备份，传给右边的 Decoder 喵