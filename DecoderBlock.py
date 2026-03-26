import torch
import torch.nn as nn
import torch.nn.functional as F
from AttentionGate import AttentionGate  # 确保这两个文件在同一目录下
from Res34Block import Res34Block

class SimpleResidual(nn.Module):
    """简单的残差块，用于解码器特征融合"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        # 如果输入输出通道不一致，用 1x1 卷积对齐维度
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x):
        return F.relu(self.conv(x) + self.shortcut(x))

class DecoderBlock(nn.Module):
    """
    改进的解码器块：
    1. 使用 ConvTranspose2d 代替 Upsample
    2. 引入 AttentionGate 进行特征筛选
    3. 使用 SimpleResidual 进行特征融合
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # --- 核心修改：转置卷积 ---
        # stride=2 实现 2 倍上采样
        # 通常 in_channels 是深层特征通道，out_channels 是浅层（skip）特征通道
        self.up_conv = nn.ConvTranspose2d(
            in_channels, 
            out_channels, 
            kernel_size=2, 
            stride=2
        )
        
        # AttentionGate：g 对应上采样后的特征，skip 对应编码器的特征
        # 注意：此时 g 的通道数已经是 out_channels 了
        self.att_gate = AttentionGate(
            F_g=out_channels,   # 上采样后的通道
            F_l=out_channels,   # 跳跃连接的通道
            F_int=out_channels // 2  # 中间层通道
        )
        
        # 拼接后的通道数为 out_channels (来自 att) + out_channels (来自 g)
        self.res_block = SimpleResidual(out_channels + out_channels, out_channels)

    def forward(self, x, skip):
        # 1. 转置卷积上采样：通道从 in_channels 变为 out_channels
        g = self.up_conv(x)
        
        # 2. 尺寸对齐检查（防止非 2^n 输入导致的 1 像素偏差）
        if g.size()[2:] != skip.size()[2:]:
            g = F.interpolate(g, size=skip.size()[2:], mode='bilinear', align_corners=True)
        
        # 3. 注意力门控处理：筛选 skip 中的有效特征
        s = self.att_gate(g, skip)
        
        # 4. 特征拼接：将注意力处理后的 skip 与 上采样特征 g 融合
        d = torch.cat([s, g], dim=1)
        
        # 5. 通过残差块输出
        return self.res_block(d)
