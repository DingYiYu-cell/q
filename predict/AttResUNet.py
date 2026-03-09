from EncoderBlock import EncoderBlock
from DecoderBlock import DecoderBlock
from Res34Block import Res34Block
import torch.nn as nn
class AttResUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.bottleneck = Res34Block(256, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        f1, p1 = self.enc1(x)
        f2, p2 = self.enc2(p1)
        f3, p3 = self.enc3(p2)
        b = self.bottleneck(p3)
        d3 = self.dec3(b, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        return self.sigmoid(self.final(d1))
