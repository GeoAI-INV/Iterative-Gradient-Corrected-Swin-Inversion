"""
Created on Thu May 30 15:03:05 2024
@author: Pang qi  o=(i+2p-f)/s+1  F=k+(k-1)(d-1)  p=(F-1)/2时,o=i/s
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange


class simam_module(torch.nn.Module):

    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = torch.tensor(e_lambda, dtype=torch.float32)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


# ----------------------------------------------------------------------------------------------------------------------
class SingleConv(nn.Module):
    """Single conv  [N,C,H,W]"""

    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=False):
        super().__init__()
        to_pad = int((kernel_size - 1) / 2)  # make sure the same size between input and output
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        if use_norm:
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=to_pad, padding_mode='reflect', bias=False),
                nn.BatchNorm2d(out_ch),
                self.activation, )
        else:
            self.single_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=to_pad, padding_mode='reflect', bias=False),
                self.activation, )

    def forward(self, x):
        return self.single_conv(x)


# ----------------------------------------------------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """double conv"""

    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True):
        super().__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        if use_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=to_pad, padding_mode='reflect', bias=False),
                nn.BatchNorm2d(out_ch),
                self.activation,
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=to_pad, padding_mode='reflect', bias=False),
                nn.BatchNorm2d(out_ch),
                self.activation)
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=to_pad, padding_mode='reflect', bias=False),
                self.activation,
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=to_pad, padding_mode='reflect', bias=False),
                self.activation)

    def forward(self, x):
        return self.double_conv(x)


# ----------------------------------------------------------------------------------------------------------------------
class Concat(nn.Module):
    """concat many tensors and the target size is the smallest or the biggest one"""

    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, *inputs):
        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]
        # if every tensor size is the smallest or the same,just cat them together
        if (np.all(np.array(inputs_shapes2) == max(inputs_shapes2)) and
                np.all(np.array(inputs_shapes3) == max(inputs_shapes3))):
            inputs_ = inputs
        else:
            target_shape2 = max(inputs_shapes2)
            target_shape3 = max(inputs_shapes3)
            inputs_ = []
            for inp in inputs:
                diffY = target_shape2 - inp.size(2)  # the third axis  Height-Y
                diffX = target_shape3 - inp.size(3)  # the last axis   Width-X
                inputs_.append(F.pad(inp, [diffX // 2, diffX - diffX // 2,
                                           diffY // 2, diffY - diffY // 2]))  # [WL,WR,HU,HD]
        return torch.cat(inputs_, dim=1)


# ----------------------------------------------------------------------------------------------------------------------
class DownBlock(nn.Module):
    """use_norm:use batchnorm or not"""

    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=True, use_att=True):
        super(DownBlock, self).__init__()
        self.use_att = use_att
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, kernel_size, use_norm))
        self.att = simam_module() if self.use_att else nn.Identity()

    def forward(self, x):
        return self.att(self.down_conv(x))


# ----------------------------------------------------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=4, kernel_size=3, use_norm=False, use_att=True):
        super(UpBlock, self).__init__()
        self.skip = skip_ch > 0
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if not self.skip:  #skip_ch为0时  not 0 => 1
            self.conv = DoubleConv(in_ch, out_ch, kernel_size, use_norm)
        else:
            self.skip_conv = SingleConv(out_ch, skip_ch, kernel_size, use_norm)
            self.conv = DoubleConv(in_ch + skip_ch, out_ch, kernel_size, use_norm)
        self.concat = Concat()
        self.att = simam_module()

    def forward(self, x1, x2):
        x = self.up(x1)
        if self.skip:
            x2 = self.skip_conv(x2)
            x = self.concat(x, x2)
        x = self.conv(x)
        return self.att(x)


# ----------------------------------------------------------------------------------------------------------------------
class UpBlock_1(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=False):
        super(UpBlock_1, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch + out_ch, out_ch, kernel_size, use_norm)
        self.concat = Concat()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = self.concat(x1, x2)
        x = self.conv(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
class UpTrans(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=False):
        super(UpTrans, self).__init__()

        self.concat = Concat()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch * 2, out_ch, kernel_size, use_norm=use_norm)

    def forward(self, x1, x2):
        x = self.conv(self.concat(x1, x2))
        x = self.up(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

    def __len__(self):
        return len(self._modules)


# ----------------------------------------------------------------------------------------------------------------------
class Regress(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Regress, self).__init__()

        self.fc1 = nn.Linear(in_ch, out_ch)
        self.activation = nn.Tanh()
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        x = rearrange(x, 'b c h w  -> b h w c')
        x = self.fc1(x)
        x = rearrange(x, 'b h w c  -> b c h w')
        x = self.activation(x)
        return x


if __name__ == '__main__':
    # torch.cuda.empty_cache()
    # net = UpTrans(32, 16)
    net = UpBlock(1, 1, skip_ch=4)
    # x = torch.randn((3, 1, 64, 128))
    x = torch.randn(10, 1, 64, 64)
    y = torch.randn(10, 1, 64, 64)
    out = net(x, y)
    print(net)
