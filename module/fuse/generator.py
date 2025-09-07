import torch
import torch.nn as nn
from torch import Tensor
from timm.models.swin_transformer import SwinTransformerBlock


class SwinAdapter(nn.Module):
    """Swin Transformer适配模块（尺寸同步）"""

    def __init__(self, dim, num_heads, window_size=7, shift_size=0, input_res=(7, 7)):
        super().__init__()
        self.window_size = window_size
        self.swin_block = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=4.0,
            input_resolution=input_res
        )
        self.swin_block.window_size = (window_size, window_size)
        self.swin_block.window_area = window_size * window_size

    def set_resolution(self, res):
        """显式更新SwinBlock输入分辨率"""
        self.swin_block.input_resolution = res

    def forward(self, x):
        feat_h, feat_w = x.shape[2], x.shape[3]
        swin_input_res = self.swin_block.input_resolution
        assert (feat_h == swin_input_res[0]) and (feat_w == swin_input_res[1]), \
            f"Swin尺寸不匹配：特征图({feat_h},{feat_w})，预期({swin_input_res[0]},{swin_input_res[1]})"

        x = x.permute(0, 2, 3, 1)
        x = self.swin_block(x)
        x = x.permute(0, 3, 1, 2)
        return x


class Generator(nn.Module):
    """U-Net++ + Swin Transformer 生成器（设备匹配修复版）"""

    def __init__(self, input_channels: int = 2, output_channels: int = 1, dim: int = 32, depth: int = 3):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.window_size = 7

        # 1. 编码器（输出通道：64, 128, 256）
        self.encoder_out_chs = []
        self.encoders = nn.ModuleList()
        for i in range(depth):
            in_ch = input_channels if i == 0 else dim * (2 ** i)
            out_ch = dim * (2 ** (i + 1))
            self.encoder_out_chs.append(out_ch)
            self.encoders.append(nn.Sequential(
                self._conv_block(in_ch, out_ch),
                SwinAdapter(dim=out_ch, num_heads=out_ch // 32, window_size=7, shift_size=i % 2 * 3),
                nn.Dropout(0.2)
            ))

        # 2. 解码器（拆分结构，最后一层输出32通道）
        self.decoders = nn.ModuleList()
        self.decoder_out_chs = []
        for i in range(depth - 1):
            in_ch = self.encoder_out_chs[depth - 1 - i]
            out_ch = dim if i == depth - 2 else self.encoder_out_chs[depth - 2 - i]
            self.decoder_out_chs.append(out_ch)
            self.decoders.append(nn.ModuleDict({
                'dropout': nn.Dropout(0.2),
                'up_conv': self._up_conv_block(in_ch, out_ch),
                'swin': SwinAdapter(dim=out_ch, num_heads=out_ch // 32, window_size=7, shift_size=i % 2 * 3)
            }))

        # 3. 密集连接块（最后一层输出32通道）
        self.dense_blocks = nn.ModuleList()
        for i in range(depth - 1):
            decoder_out_ch = self.decoder_out_chs[i]
            encoder_level = depth - 2 - i
            encoder_chs_sum = sum(self.encoder_out_chs[encoder_level:])
            in_ch = decoder_out_ch + encoder_chs_sum
            out_ch = dim if i == depth - 2 else self.encoder_out_chs[encoder_level]
            self.dense_blocks.append(self._conv_block(in_ch, out_ch))

        # 4. 跨模态注意力门控 + 新增：IR/VI通道扩展卷积（预定义为模型属性）
        self.cross_attn_gates = nn.ModuleList()
        self.ir_expanders = nn.ModuleList()  # IR单通道扩展到编码器通道
        self.vi_expanders = nn.ModuleList()  # VI单通道扩展到编码器通道
        for ch in self.encoder_out_chs:
            self.cross_attn_gates.append(nn.Sequential(
                nn.Conv2d(2 * ch, ch, 1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 1),
                nn.Sigmoid()
            ))
            self.ir_expanders.append(nn.Conv2d(1, ch, kernel_size=1))  # 预定义IR扩展层
            self.vi_expanders.append(nn.Conv2d(1, ch, kernel_size=1))  # 预定义VI扩展层

        # 5. 输出层（输入32通道）
        self.out_conv = nn.Conv2d(dim, output_channels, 1)
        self.tanh = nn.Tanh()

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def _up_conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, ir: Tensor, vi: Tensor) -> Tensor:
        # 1. 编码器特征提取
        x = torch.cat([ir, vi], dim=1)
        skip_connections = []
        skip_resolutions = []
        for i in range(self.depth):
            # 尺寸对齐（7的倍数）
            current_h, current_w = x.shape[2], x.shape[3]
            target_h = ((current_h + 6) // 7) * 7
            target_w = ((current_w + 6) // 7) * 7
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = nn.Upsample(size=(target_h, target_w), mode='bilinear', align_corners=True)(x)

            # 更新Swin分辨率并处理
            self.encoders[i][1].set_resolution((target_h, target_w))
            x = self.encoders[i](x)
            skip_connections.append(x)
            skip_resolutions.append((target_h, target_w))

            # 下采样
            if i < self.depth - 1:
                x = nn.MaxPool2d(2, 2)(x)

        # 2. 跨模态融合（使用预定义的扩展层，确保设备一致）
        fused_skips = []
        for i, (feat, gate, ir_expand, vi_expand) in enumerate(zip(
                skip_connections, self.cross_attn_gates, self.ir_expanders, self.vi_expanders
        )):
            ch = feat.shape[1]
            # 使用预定义的卷积层扩展通道（已随模型移动到正确设备）
            ir_feat = ir_expand(ir)  # 替代动态创建的nn.Conv2d
            vi_feat = vi_expand(vi)  # 替代动态创建的nn.Conv2d
            # 尺寸对齐
            ir_feat = nn.Upsample(size=feat.shape[2:], mode='bilinear', align_corners=True)(ir_feat)
            vi_feat = nn.Upsample(size=feat.shape[2:], mode='bilinear', align_corners=True)(vi_feat)
            # 注意力融合
            attn = gate(torch.cat([ir_feat, vi_feat], dim=1))
            fused_skips.append(ir_feat * attn + vi_feat * (1 - attn))

        # 3. 解码器特征恢复
        current_feat = fused_skips[-1]
        for i in range(len(self.decoders)):
            decoder = self.decoders[i]
            # 分步处理：Dropout → 上采样 → Swin → 密集连接
            current_feat = decoder['dropout'](current_feat)
            current_feat = decoder['up_conv'](current_feat)
            up_res = (current_feat.shape[2], current_feat.shape[3])

            # 更新Swin分辨率
            decoder['swin'].set_resolution(up_res)
            current_feat = decoder['swin'](current_feat)

            # 多尺度融合
            encoder_level = self.depth - 2 - i
            dense_feats = [current_feat]
            for j in range(encoder_level, self.depth):
                high_feat = fused_skips[j]
                if high_feat.shape[2:] != up_res:
                    high_feat = nn.Upsample(size=up_res, mode='bilinear', align_corners=True)(high_feat)
                dense_feats.append(high_feat)

            # 密集连接
            current_feat = torch.cat(dense_feats, dim=1)
            current_feat = self.dense_blocks[i](current_feat)

        # 4. 输出融合图像
        return self.tanh(self.out_conv(current_feat))


# 验证代码
if __name__ == "__main__":
    # 测试设备一致性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ir = torch.randn(2, 1, 459, 622).to(device)
    vi = torch.randn(2, 1, 459, 622).to(device)
    model = Generator(dim=32, depth=3).to(device)  # 模型移动到设备
    output = model(ir, vi)
    print(f"输出形状: {output.shape}，设备: {output.device}")  # 应显示cuda
    print("模型测试通过！")
