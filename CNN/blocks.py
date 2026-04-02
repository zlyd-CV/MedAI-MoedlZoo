import torch
import torch.nn as nn
from typing import List, Tuple

class DoubleConv(nn.Module):
    """
    标准的双层卷积块 (Conv2d -> BatchNorm2d -> ReLU) * 2
    
    Args:
        in_channels (int): 输入特征图的通道数
        out_channels (int): 输出特征图的通道数
        mid_channels (int, optional): 中间层的通道数。如果不指定，则默认等于 out_channels
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)