from torch import nn, Tensor


class Discriminator(nn.Module):
    """
    Use to discriminate fused images and source images.
    """

    def __init__(self, dim: int = 64, size: tuple[int, int] = (320, 320)):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            # ===== 谱归一化卷积 =====
            nn.utils.spectral_norm(nn.Conv2d(1, dim * 4, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(dim * 4, dim * 8, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            # ===== 全局池化替代全连接 =====
            nn.AdaptiveAvgPool2d(1)
        )
        self.linear = nn.utils.spectral_norm(nn.Linear(dim * 8, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)         # 输出形状: [B, C, 1, 1]
        x = x.view(x.size(0), -1)  # 展平为[B, C]
        x = self.linear(x)
        return x
