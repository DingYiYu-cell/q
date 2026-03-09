import torch
import torch.nn as nn
from torchvision import models
# 保持你原有的导入，但 EncoderBlock 会被 ResNet 替换喵
from DecoderBlock import DecoderBlock 
from CBAM import CBAM # 记得把你刚才写的 CBAM 类保存在 CBAM.py 里喵

class AttResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # 1. 加载 ResNet34 引擎 (不使用预训练则 pretrained=False) 喵
        resnet = models.resnet34(pretrained=False) 
        
        # 处理单通道输入：如果主人图是灰色的，要把第一层改了喵
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu
        )
        self.pool = resnet.maxpool # 下采样用喵
        
        # 2. 提取 ResNet 的四个阶段作为 Encoder 喵
        self.enc1 = resnet.layer1 # 输出 64 通道
        self.enc2 = resnet.layer2 # 输出 128 通道
        self.enc3 = resnet.layer3 # 输出 256 通道
        self.enc4 = resnet.layer4 # 输出 512 通道 (这一层就是 Bottleneck 喵)

        # 3. 在瓶颈处插入 CBAM 喵！
        self.cbam = CBAM(512) 

        # 4. 解码器部分 (通道数要和 ResNet 对应上喵)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        # 最后的输出层喵
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # --- Encoder 阶段 ---
        x0 = self.init_conv(x)      # 1/2 大小 (64通道)
        x1 = self.enc1(self.pool(x0)) # 1/4 大小 (64通道)
        x2 = self.enc2(x1)          # 1/8 大小 (128通道)
        x3 = self.enc3(x2)          # 1/16 大小 (256通道)
        
        # --- Bottleneck + CBAM 阶段 ---
        b = self.enc4(x3)           # 1/32 大小 (512通道)
        b = self.cbam(b)            # 经过注意力强化喵！
        
        # --- Decoder 阶段 (带跳跃连接) ---
        d3 = self.dec3(b, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        
        # 因为 ResNet 第一层有缩放，最后可能需要额外上采样一次还原原图喵
        out = self.final_up(d1)
        return self.sigmoid(self.final(out))