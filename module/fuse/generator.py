import torch
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    r"""最终修正版U-Net++生成器，解决解码器循环顺序错误导致的尺寸不匹配"""

    def __init__(self, input_channels: int = 2, output_channels: int = 1, dim: int = 32, depth: int = 4):
        super(Generator, self).__init__()
        self.depth = depth
        self.init_features = dim

        # 编码器块（无池化）
        self.encoders = nn.ModuleList()
        for i in range(depth):
            in_ch = input_channels if i == 0 else dim * (2 ** (i - 1))
            out_ch = dim * (2 ** i)
            self.encoders.append(self._conv_block(in_ch, out_ch))

        # 解码器上采样块（按深层到浅层顺序定义：2→1→0）
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = dim * (2 ** (i + 1))  # 输入：上一层解码器输出通道
            out_ch = dim * (2 ** i)  # 输出：当前解码器通道
            self.decoders.append(self._up_conv_block(in_ch, out_ch))

        # 密集连接融合块（按深层到浅层顺序定义：2→1→0）
        self.dense_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            in_ch = dim * (2 ** i)  # 基础通道：解码器输出通道
            for j in range(i + 1, depth):  # 添加深层编码器特征通道
                in_ch += dim * (2 ** j)
            self.dense_blocks.append(self._conv_block(in_ch, dim * (2 ** i)))

        # 输出层
        self.out_conv = nn.Conv2d(dim, output_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _up_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, ir: Tensor, vi: Tensor) -> Tensor:
        x = torch.cat([ir, vi], dim=1)  # [B, 2, 224, 224]

        # 编码器特征存储（尺寸从大到小：224→112→56→28）
        skip_connections = []
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            if len(skip_connections) < self.depth:
                x = nn.MaxPool2d(2, 2)(x)

        # 初始特征：最深层编码器输出（28x28, 256通道）
        current_features = skip_connections[-1]

        # 解码器循环（从深层到浅层：2→1→0）
        for decoder_idx, decoder_level in enumerate(range(self.depth - 2, -1, -1)):
            current_encoder_level = decoder_level  # 2,1,0

            # 1. 上采样当前特征
            current_features = self.decoders[decoder_idx](current_features)

            # 2. 收集并上采样路径特征
            dense_features = [current_features]
            for j in range(current_encoder_level + 1, self.depth):
                upsample_times = j - current_encoder_level
                path_feature = skip_connections[j]

                # 上采样至当前层级尺寸
                for _ in range(upsample_times):
                    path_feature = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)(path_feature)

                dense_features.append(path_feature)

            # 3. 拼接融合
            current_features = torch.cat(dense_features, dim=1)
            current_features = self.dense_blocks[decoder_idx](current_features)

        return self.tanh(self.out_conv(current_features))