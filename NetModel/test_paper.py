import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from NetModel.Net2D.block2d import Regress, OutConv
from NetModel.MyTransformer.TransBlock import ViT, Transformer, transencoder


class SingleFeature(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        f = kernel_size + (kernel_size - 1) * (dilation - 1)
        to_pad = int((f - 1) / 2)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.feature = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, to_pad), dilation=(1, dilation),
                      padding_mode='reflect', bias=False),
            # nn.GroupNorm(2, out_ch),  # 会造成梯度爆炸 loss值一直不变
            self.activation)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 防止过拟合

    def forward(self, x):
        return self.maxpool(self.feature(x))

class ConvEncoder(nn.Module):
    def __init__(self, dim, kernel_size, mlp_ratio=4, dropout=0.):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2),
                              padding_mode='reflect', bias=False, groups=dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w  -> b (h w) c')
        x = self.net(x)
        x = rearrange(x, 'b (h w) c  -> b c h w', h=h, w=w)
        x = x + shortcut
        return x


class MulConv(nn.Module):
    def __init__(self, dim, input_size, patch_size, out_ch=None, heads=4, depth=1, kernel_size=[3, 5, 7], dropout=0.0):
        super().__init__()
        if out_ch is None:
            out_ch = dim
        self.net = nn.Sequential(ConvEncoder(dim, kernel_size[0]),
                                 ConvEncoder(dim, kernel_size[1]),
                                 ConvEncoder(dim, kernel_size[2]))
        # self.trans = ViT(dim, input_size, patch_size, heads, depth=depth)
        self.trans = transencoder(dim, input_size, patch_size, heads, depth=depth)

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x = self.net(x)
        x = self.trans(x)
        x = x + shortcut
        return x

class CLWTNet(nn.Module):
    def __init__(self, in_ch, input_size, patch_size, head=4, embedd_dim=8, dilation=[1, 3, 6], depth=2):
        super().__init__()
        self.d1 = SingleFeature(in_ch, embedd_dim, kernel_size=5, dilation=dilation[0])
        self.d2 = SingleFeature(in_ch, embedd_dim, kernel_size=5, dilation=dilation[1])
        self.d3 = SingleFeature(in_ch, embedd_dim, kernel_size=5, dilation=dilation[2])
        self.mulconv1 = MulConv(embedd_dim, input_size, patch_size, heads=head, depth=depth)
        self.mulconv2 = MulConv(embedd_dim * 3, input_size, patch_size, heads=head*3, depth=depth)
        self.mulconv3 = MulConv(16, input_size, patch_size, heads=head*2, depth=depth)
        self.conv1 = nn.Conv2d(embedd_dim * 3, 16, kernel_size=3, padding=1, padding_mode='reflect', bias=False)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1,  padding_mode='reflect', bias=False)
        self.out_p = Regress(16, 1)
        # self.out_p = OutConv(16,1)
        
    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        mul1 = self.mulconv1(x1)
        mul2 = self.mulconv1(x2)
        mul3 = self.mulconv1(x3)
        x_end1 = torch.cat([x1, x2, x3], dim=1)
        x_end2 = torch.cat([mul1, mul2, mul3], dim=1)

        x_end1 = self.mulconv2(x_end1)
        x_end2 = x_end2 + x_end1
        x_end2 = self.mulconv2(x_end2)
        x_end2 = self.conv1(x_end2)
        x_end2 = self.mulconv3(x_end2)
        # x_end2 = self.conv2(x_end2)
        # scores = self.mulconv3(x_end2)
        return self.out_p(x_end2)


if __name__ == '__main__':
    net = CLWTNet(1, 40, 4)
    x = torch.randn(10, 1, 40, 40)  
    output = net(x)

