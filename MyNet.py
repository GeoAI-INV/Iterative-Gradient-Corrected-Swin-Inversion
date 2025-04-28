"""
Created on Thu May 30 16:56:01 2024
@author: Pang Qi
"""
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange
from NetModel.Net2D import block2d
from NetModel.TransformerNet.transblock import TransBlock, SwinBlock
from NetModel.MyTransformer.TransBlock import ViT
from NetModel.efficient_kan import KANLinear

from timm.models.layers import trunc_normal_
import math


class SingleFeature(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        f = kernel_size + (kernel_size - 1) * (dilation - 1)
        to_pad = int((f - 1) / 2)
        self.feature = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=to_pad, dilation=dilation, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(out_ch))
        self.maxpool = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect', bias=True)

    def forward(self, x):
        return self.maxpool(self.feature(x))


class MultiFeature(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch=8, kernel_size=5, dilation=[1, 3, 6]):
        super().__init__()
        self.layers = len(dilation)
        self.muti_f = nn.ModuleList([SingleFeature(in_ch, hidden_ch, kernel_size, dilation[i])
                                     for i in range(self.layers)])
        self.concat = block2d.Concat()
        self.conv = block2d.DoubleConv(hidden_ch * self.layers, out_ch, use_norm=True)

    def forward(self, x):
        out_f = []
        for get_feature in self.muti_f:
            out_f.append(get_feature(x))
        scores = self.concat(*out_f)
        return self.conv(scores)


class UNet2D(nn.Module):  #在之前的论文中 每个下采样加入了能力聚集注意力模块  # 在论文中是加入了use_att
    def __init__(self, in_ch, out_ch, down_channels, layers, skip_channels,
                 use_sigmoid=False, use_norm=False, use_att=True):
        super(UNet2D, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = block2d.SingleConv(in_ch, down_channels, use_norm=use_norm)
        for i in range(layers):
            down_in_ch = down_channels * 2 ** i
            down_out_ch = down_channels * 2 ** (i + 1)
            up_in_ch = down_channels * 2 ** (layers - i)
            up_out_ch = down_channels * 2 ** (layers - i - 1)
            self.down.append(block2d.DownBlock(down_in_ch, down_out_ch, use_norm=use_norm, use_att=use_att))
            self.up.append(
                block2d.UpBlock(up_in_ch, up_out_ch, skip_ch=skip_channels, use_norm=use_norm, use_att=use_att))  # skip_channels[-(i+1)]
        self.outc = block2d.Regress(in_ch=down_channels, out_ch=out_ch)

    def forward(self, x, prompt=None):
        if prompt is not None:
            x = torch.cat([x, prompt], dim=1)
        x_down = [self.inc(x)]  # save high resolution-downsamples
        for downsample in self.down:
            x_down.append(downsample(x_down[-1]))
        scores = x_down[-1]
        for i, upsample in enumerate(self.up):
            scores = upsample(scores, x_down[-i - 2])
        return self.outc(scores)


class UNet2D_all(nn.Module):
    def __init__(self, in_ch, out_ch, down_channels, layers, skip_channels,
                 use_sigmoid=False, use_norm=False, use_att=True):
        super(UNet2D_all, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = block2d.SingleConv(in_ch, down_channels, use_norm=use_norm)
        for i in range(layers):
            down_in_ch = down_channels * 2 ** i
            down_out_ch = down_channels * 2 ** (i + 1)
            up_in_ch = down_channels * 2 ** (layers - i)
            up_out_ch = down_channels * 2 ** (layers - i - 1)
            self.down.append(block2d.DownBlock(down_in_ch, down_out_ch, use_norm=use_norm, use_att=use_att))
            self.up.append(
                block2d.UpBlock(up_in_ch, up_out_ch, skip_ch=skip_channels, use_norm=use_norm, use_att=use_att))  # skip_channels[-(i+1)]
        self.outc = block2d.Regress(in_ch=down_channels, out_ch=out_ch)

    def forward(self, x, prompt=None):
        if prompt is not None:
            x = torch.cat([x, prompt], dim=1)
        x_down = [self.inc(x)]  # save high resolution-downsamples
        x_up = []
        for downsample in self.down:
            x_down.append(downsample(x_down[-1]))
        scores = x_down[-1]
        for i, upsample in enumerate(self.up):
            scores = upsample(scores, x_down[-i - 2])
            x_up.append(scores)
        return self.outc(scores), x_down[-3:], x_up

class TUNet2D(nn.Module):
    def __init__(self, in_ch, out_ch, down_channels, layers, skip_channels, num_heads, trans_num=8,
                 use_sigmoid=False, use_norm=False):
        super(TUNet2D, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = block2d.SingleConv(in_ch * 2, down_channels, use_norm=use_norm)

        for i in range(layers):
            down_in_ch = down_channels * 2 ** i
            down_out_ch = down_channels * 2 ** (i + 1)
            up_in_ch = down_channels * 2 ** (layers - i)
            up_out_ch = down_channels * 2 ** (layers - i - 1)
            self.down.append(block2d.DownBlock(down_in_ch, down_out_ch, use_norm=use_norm))
            self.up.append(
                block2d.UpBlock(up_in_ch, up_out_ch, skip_ch=skip_channels, use_norm=use_norm))  #skip_channels[-(i+1)]

        final_dim = down_channels * 2 ** layers

        self.translayer = nn.Sequential(*[TransBlock(final_dim, num_heads) for _ in range(trans_num)])

        self.outc = block2d.OutConv(in_ch=down_channels, out_ch=out_ch)

    def forward(self, x, y):
        x_down = [self.inc(torch.cat((x, y), dim=1))]  #save high resolution-downsamples

        for downsample in self.down:
            x_down.append(downsample(x_down[-1]))
        scores = x_down[-1]

        scores = self.swinlayer(scores)

        for i, upsample in enumerate(self.up):
            scores = upsample(scores, x_down[-i - 2])

        return torch.sigmoid(self.outc(scores)) if self.use_sigmoid else self.outc(scores)


class TUNet2D_1(nn.Module):
    def __init__(self, in_ch, out_ch, down_channels, num_heads, drop_path=0., trans_num=8,
                 use_sigmoid=False, use_norm=True):
        super(TUNet2D_1, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        # self.inc = block2d.SingleConv(in_ch * 2, down_channels, use_norm=use_norm)
        self.inc = block2d.SingleConv(in_ch, down_channels, use_norm=use_norm)

        self.down1 = block2d.DownBlock(down_channels, down_channels * 2, use_norm=use_norm)
        self.down2 = block2d.DownBlock(down_channels * 2, down_channels * 4, use_norm=use_norm)
        self.down3 = block2d.DownBlock(down_channels * 4, down_channels * 8, use_norm=use_norm)
        # self.translayer = nn.Sequential(*[TransBlock(down_channels * 8, num_heads,drop_path=drop_path) for _ in range(trans_num)])
        self.translayer = nn.Sequential(
            ViT(in_channels=down_channels * 8, patch_size=4, heads=num_heads, mlp_ratio=4, depth=trans_num,
                dropout=drop_path),
            block2d.SingleConv(down_channels * 8, down_channels * 8, kernel_size=1, use_norm=use_norm))
        # self.up1 = block2d.UpTrans(down_channels * 8, down_channels * 4)
        self.up1 = block2d.UpBlock_1(down_channels * 8, down_channels * 4, use_norm=use_norm)
        self.up2 = block2d.UpBlock_1(down_channels * 4, down_channels * 2, use_norm=use_norm)
        self.up3 = block2d.UpBlock_1(down_channels * 2, down_channels, use_norm=use_norm)

        self.outc = block2d.Regress(in_ch=down_channels, out_ch=out_ch)
        # self.outc = block2d.OutConv(in_ch=down_channels, out_ch=out_ch)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # x_down = self.inc(torch.cat((x, y), dim=1))
        x_down = self.inc(x)
        x_down1 = self.down1(x_down)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        x_trans = self.translayer(x_down3)

        x_up1 = self.up1(x_trans, x_down2)
        x_up2 = self.up2(x_up1, x_down1)
        x_up3 = self.up3(x_up2, x_down)
        scores = self.outc(x_up3)

        return scores


class SUNet2D_1(nn.Module):
    def __init__(self, in_ch, out_ch, down_channels, num_heads, drop_path=0., trans_num=8,
                 use_sigmoid=False, use_norm=True):
        super(SUNet2D_1, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        # self.inc = block2d.SingleConv(in_ch * 2, down_channels, use_norm=use_norm)
        self.inc = block2d.SingleConv(in_ch, down_channels, use_norm=use_norm)

        self.down1 = block2d.DownBlock(down_channels, down_channels * 2, use_norm=use_norm)
        self.down2 = block2d.DownBlock(down_channels * 2, down_channels * 4, use_norm=use_norm)
        self.down3 = block2d.DownBlock(down_channels * 4, down_channels * 8, use_norm=use_norm)
        head_dim1 = (down_channels * 8) // num_heads
        self.swinlayer = nn.Sequential(
            *[SwinBlock(down_channels * 8, head_dim1, drop_path=drop_path, type='W' if not i % 2 else 'SW') for i in
              range(trans_num)],
            block2d.SingleConv(down_channels * 8, down_channels * 8, kernel_size=1, use_norm=use_norm)
        )

        self.up1 = block2d.UpBlock_1(down_channels * 8, down_channels * 4, use_norm=use_norm)
        # self.up1 = block2d.UpTrans(down_channels * 8, down_channels * 4, use_norm=use_norm)
        self.up2 = block2d.UpBlock_1(down_channels * 4, down_channels * 2, use_norm=use_norm)
        self.up3 = block2d.UpBlock_1(down_channels * 2, down_channels, use_norm=use_norm)
        # head_dim2 = down_channels // 2
        # self.low_layer = nn.Sequential(
        #     *[SwinBlock(down_channels, head_dim2, drop_path=drop_path, type='W' if not i % 2 else 'SW') for i in
        #       range(2)])
        self.outc = block2d.Regress(in_ch=down_channels, out_ch=out_ch)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x_down = self.inc(x)
        x_down1 = self.down1(x_down)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)

        x_trans = self.swinlayer(x_down3)
        x_up1 = self.up1(x_trans, x_down2)
        x_up2 = self.up2(x_up1, x_down1)
        x_up3 = self.up3(x_up2, x_down)

        scores = self.outc(x_up3)

        return scores


class UNet2D_AVO(nn.Module):
    def __init__(self, in_ch, out_ch, down_channels, layers, skip_channels,
                 use_sigmoid=False, use_norm=False):
        super(UNet2D_AVO, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.inc = block2d.SingleConv(in_ch, down_channels, use_norm=use_norm)
        for i in range(layers):
            down_in_ch = down_channels * 2 ** i
            down_out_ch = down_channels * 2 ** (i + 1)
            up_in_ch = down_channels * 2 ** (layers - i)
            up_out_ch = down_channels * 2 ** (layers - i - 1)
            self.down.append(block2d.DownBlock(down_in_ch, down_out_ch, use_norm=use_norm))
            self.up.append(
                block2d.UpBlock(up_in_ch, up_out_ch, skip_ch=skip_channels, use_norm=use_norm))  # skip_channels[-(i+1)]
        self.outc = block2d.Regress(in_ch=down_channels, out_ch=out_ch)

    def forward(self, x):
        x_down = [self.inc(x)]  # save high resolution-downsamples
        for downsample in self.down:
            x_down.append(downsample(x_down[-1]))
        scores = x_down[-1]
        for i, upsample in enumerate(self.up):
            scores = upsample(scores, x_down[-i - 2])
        # scores = torch.cat((self.outc_vp(scores), self.outc_vs(scores), self.outc_den(scores)),dim=1)
        return self.outc(scores)


class KanLayer(nn.Module):
    "equal to MLP"
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = out_features
        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        self.kan1 = KANLinear(
            in_features,
            hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range, )

        self.kan2 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range, )

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> (b h w) c')
        x = self.kan1(x)
        x = self.kan2(x)
        x = rearrange(x, '(b h w) c -> b c h w', b=B, h=H, w=W, c=self.dim)
        return x


class ForwardNet(nn.Module):
    def __init__(self, in_ch, out_ch1, out_ch, hidden_ch=32, use_sigmoid=False, use_norm=False):
        super(ForwardNet, self).__init__()

        self.multi = MultiFeature(in_ch, out_ch1)
        self.kan = KanLayer(out_ch1, hidden_ch, out_ch)
        self.outc = block2d.Regress(in_ch=out_ch, out_ch=out_ch)

    def forward(self, x):
        scores = self.multi(x)
        return self.kan(scores)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    x = torch.randn(10, 1, 551, 1361)
    # y = torch.randn(10, 1, 256, 256)
    x = torch.randn(1, 1, 400, 2078)
    net = UNet2D(in_ch=1, out_ch=1, down_channels=16, layers=3, skip_channels=4, use_norm=False, use_att=False)
    out = net(x)
    # print(x.shape)
