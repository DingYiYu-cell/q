class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 修正：这里直接传参数，不再使用不匹配的关键词 gating_channels
        self.att_gate = AttentionGate(in_channels, out_channels, out_channels // 2)
        self.res_block = ResBlock(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        g = self.upsample(x)
        # 修正：尺寸检查逻辑
        if g.size()[2:] != skip.size()[2:]:
            g = F.interpolate(g, size=skip.size()[2:], mode='bilinear', align_corners=True)
        
        s = self.att_gate(g, skip) # 修正：直接传参
        d = torch.cat([s, g], dim=1)
        return self.res_block(d)