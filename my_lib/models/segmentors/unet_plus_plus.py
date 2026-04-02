import torch
import torch.nn as nn

from my_lib.modules.conv_blocks import DoubleConv

class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested U-Net) 架构
    支持 Deep Supervision (深度监督)。
    
    References:
        "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
        https://arxiv.org/abs/1807.10165
        
    Args:
        in_channels (int): 输入图像的通道数
        num_classes (int): 分割的类别数
        base_filters (int): 初始基础特征通道数，默认为 64
        deep_supervision (bool): 是否启用深度监督。若启用，将返回多个尺度的输出列表。
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_filters: int = 64, deep_supervision: bool = False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        nb_filter = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 16]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 通常 UNet++ 库倾向于使用双线性插值减少棋盘效应
        
        # 编码器部分 (Encoder / Backbone, 对应原论文的 j=0 列)
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # 密集跳跃连接部分 (Dense Skip Pathways)
        # 对应原论文的 j=1 列 (从不同尺度的特征图进行第一次融合)
        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        # 对应原论文的 j=2 列 (从不同尺度的特征图进行第二次融合)
        self.conv0_2 = DoubleConv(nb_filter[0]*2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2 + nb_filter[3], nb_filter[2])

        # 对应原论文的 j=3 列 (从不同尺度的特征图进行第三次融合)
        self.conv0_3 = DoubleConv(nb_filter[0]*3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3 + nb_filter[2], nb_filter[1])

        # 对应原论文的 j=4 列 (从不同尺度的特征图进行第四次融合)
        self.conv0_4 = DoubleConv(nb_filter[0]*4 + nb_filter[1], nb_filter[0])

        # 最终输出层
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # 编码器 (Encoder / Backbone, 对应原论文的 j=0 列)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # 密集跳跃连接部分 (Dense Skip Pathways)
        # 第一阶段特征融合 (对应原论文的 j=1 列)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # 第二阶段特征融合 (对应原论文的 j=2 列)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        # 第三阶段特征融合 (对应原论文的 j=3 列)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # 第四阶段特征融合 (对应原论文的 j=4 列)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # 输出处理
        if self.deep_supervision:
            # 返回多尺度预测，在训练时计算各尺度的 Loss 并求均值 (或者加权)，可以加速收敛
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final_conv(x0_4)

def test_unet_plus_plus():
    """使用 torchinfo 打印 UNet++ 模型结构信息"""
    try:
        from torchinfo import summary
    except ImportError:
        print("请先安装 torchinfo: pip install torchinfo")
        return

    print("========= 测试 UNet++ (开启 Deep Supervision) =========")
    model = UNetPlusPlus(in_channels=3, num_classes=1, base_filters=64, deep_supervision=True)
    # 假设输入的是单张 3 通道 256x256 的图像 (B, C, H, W)
    summary(model, input_size=(1, 3, 256, 256), device="cpu")

if __name__ == "__main__":
    test_unet_plus_plus()
