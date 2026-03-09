import torch
import torch.nn as nn

# 这是一个精简好用的 CBAM 模块喵！
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 1. 通道注意力 (Channel Attention) 喵
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 2. 空间注意力 (Spatial Attention) 喵
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 先做通道加权喵
        x = x * self.ca(x)
        # 再做空间加权喵（利用最大池化和平均池化的拼接）
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weight = self.sa(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_weight

# 在你的模型初始化里这么用喵：
# self.bottleneck_cbam = CBAM(512)