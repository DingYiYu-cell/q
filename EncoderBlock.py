from ResBlock import ResBlock
import torch.nn as nn
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        f = self.res_block(x)
        p = self.pool(f)
        return f, p