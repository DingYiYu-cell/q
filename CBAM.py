import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 共享卷积核
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()#!这里单独写，避免两个并行的通道注意力方法重复计算sigmoid发生错误
        
        # 空间注意力
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        #通道注意力（并行计算）
        avg_cout = self.shared_conv(nn.AdaptiveAvgPool2d(1)(x))
        max_cout = self.shared_conv(nn.AdaptiveMaxPool2d(1)(x))
        ca_weight = self.sigmoid(avg_cout + max_cout)
        x = x * ca_weight
        avg_sout = torch.mean(x, dim=1, keepdim=True)
        max_sout = torch.max(x, dim=1, keepdim=True)[0]
        sa_weight = self.sa(torch.cat([avg_sout, max_sout], dim=1))#拼接两个权重通道
        return x * sa_weight
        

# 初始化
# self.bottleneck_cbam = CBAM(512)