import torch
import torch.nn as nn

# 假设 DoubleConv 被定义在 my_lib/modules/conv_blocks 中
# 这里使用相对导入将其引入
from my_lib.modules.conv_blocks import DoubleConv

class UNet(nn.Module):
    """
    标准的 U-Net 架构
    
    References:
        "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        https://arxiv.org/abs/1505.04597
        
    Args:
        in_channels (int): 输入图像的通道数 (例如 RGB 为 3，灰度为 1)
        num_classes (int): 分割的类别数 (二分类通常设为 1，多分类设为 N)
        base_filters (int): 初始基础特征通道数，默认为 64
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_filters: int = 64):
        super().__init__()
        
        filters = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 16]
        
        # 编码器 (Downsampling)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = DoubleConv(in_channels, filters[0])
        self.down2 = DoubleConv(filters[0], filters[1])
        self.down3 = DoubleConv(filters[1], filters[2])
        self.down4 = DoubleConv(filters[2], filters[3])
        
        # 瓶颈层 (Bottleneck)
        self.bottleneck = DoubleConv(filters[3], filters[4])
        
        # 解码器 (Upsampling)
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.conv4 = DoubleConv(filters[4], filters[3]) # skip connection 后通道翻倍，所以输入是 filters[4]
        
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.conv3 = DoubleConv(filters[3], filters[2])
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.conv2 = DoubleConv(filters[2], filters[1])
        
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.conv1 = DoubleConv(filters[1], filters[0])
        
        # 最终输出层 (输出未归一化的 logits)
        self.final_conv = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        conv1 = self.down1(x)
        conv2 = self.down2(self.pool(conv1))
        conv3 = self.down3(self.pool(conv2))
        conv4 = self.down4(self.pool(conv3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(conv4))
        
        # Decoder with Skip Connections
        dec4 = self.up4(bottleneck)
        dec4 = torch.cat([conv4, dec4], dim=1)
        dec4 = self.conv4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([conv3, dec3], dim=1)
        dec3 = self.conv3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([conv2, dec2], dim=1)
        dec2 = self.conv2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([conv1, dec1], dim=1)
        dec1 = self.conv1(dec1)
        
        logits = self.final_conv(dec1)
        # 注意：此处不返回 sigmoid/softmax，由外部的 Loss 函数处理
        return logits

def test_unet():
    """使用 torchinfo 打印 U-Net 模型结构信息"""
    try:
        from torchinfo import summary
    except ImportError:
        print("请先安装 torchinfo: pip install torchinfo")
        return

    print("========= 测试 U-Net =========")
    model = UNet(in_channels=3, num_classes=1, base_filters=64)
    # 假设输入的是单张 3 通道 256x256 的图像 (B, C, H, W)
    summary(model, input_size=(1, 3, 256, 256), device="cpu")

if __name__ == "__main__":
    test_unet()
