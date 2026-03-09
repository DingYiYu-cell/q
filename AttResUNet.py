import torch
import torch.nn as nn
import torch.nn.functional as F
from Res34Block import Res34Block # 确保这个文件里有你改好的 stride=1
from DecoderBlock import DecoderBlock 

class AttResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        # 1. 直接使用你封装好的带 CBAM 的 Encoder
        self.encoder = Res34Block(pretrained=False) # 内部已处理 stride=1 和 1通道输入
        
        # 2. 解码器部分 - 必须有 4 个阶段才能对应 x4, x3, x2, x1, x0
        self.dec4 = DecoderBlock(512, 256) # 处理 x4 -> x3
        self.dec3 = DecoderBlock(256, 128) # 处理 x3 -> x2
        self.dec2 = DecoderBlock(128, 64)  # 处理 x2 -> x1
        self.dec1 = DecoderBlock(64, 64)   # 处理 x1 -> x0 (这一层让尺寸从 112 回到 224)
        
        # 最后的输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder 阶段 ---
        # 拿到你在 Res34Block 里准备好的 5 个特征备份
        # x0:224, x1:112, x2:56, x3:28, x4:14
        x0, x1, x2, x3, x4 = self.encoder(x) 
        
        # --- Decoder 阶段 ---
        d4 = self.dec4(x4, x3)   # 输出 28x28
        d3 = self.dec3(d4, x2)   # 输出 56x56
        d2 = self.dec2(d3, x1)   # 输出 112x112
        d1 = self.dec1(d2, x0)   # 输出 224x224 (对接最精细的 x0 层)
        
        # --- 输出阶段 ---
        out = self.final_conv(d1)
        
        # 重点：只返回一个 Tensor，并且在这里做 Sigmoid
        # 确保没有多余的逗号喵！
        return torch.sigmoid(out)