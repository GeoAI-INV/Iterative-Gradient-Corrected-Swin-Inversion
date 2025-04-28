import math
import torch
import torch.nn as nn
import numpy as np
from thop import profile
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
from NetModel.Net2D.block2d import DoubleConv, Concat, SingleConv
from NetModel.Net2D.svt import FFN

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
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  #防止过拟合

    def forward(self, x):
        return self.maxpool(self.feature(x))


class MultiFeature(nn.Module):
    def __init__(self, in_ch, out_ch, out_dim, kernel_size=5, dilation=[1, 3, 6]):
        super().__init__()
        self.layers = len(dilation)
        self.muti_f = nn.ModuleList([SingleFeature(in_ch, out_ch, kernel_size, dilation[i])
                                     for i in range(self.layers)])
        self.concat = Concat()
        self.conv = DoubleConv(out_ch * self.layers, out_dim, use_norm=True)

    def forward(self, x):
        out_f = []
        for get_feature in self.muti_f:
            out_f.append(get_feature(x))
        scores = self.concat(*out_f)
        return self.conv(scores)
    
class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type  #w代表窗内自注意力  sw代表窗外移位
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)  #create q k v

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(2 * window_size - 1, 2 * window_size - 1, self.n_heads).transpose(1,
                                                                                                                 2).transpose(
                0, 1))
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

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type != 'W': x = torch.roll(x, shifts=(-(self.window_size // 2), -(self.window_size // 2)), dims=(1, 2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size // 2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type != 'W': output = torch.roll(output, shifts=(self.window_size // 2, self.window_size // 2),
                                                 dims=(1, 2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        # negative is allowed
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]


class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(SwinBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        # print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)

        self.mlp = FFN(input_dim, 4 * input_dim)

        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, 4 * input_dim),
        #     nn.GELU(),
        #     nn.Linear(4 * input_dim, output_dim),
        # )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SwinConvBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block  (B,C,H,W)->(B,C,H,W)
        """
        super(SwinConvBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution

        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:  #防止图片太小 很小时采用窗内
            self.type = 'W'

        self.trans_block = SwinBlock(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path,
                                 self.type, self.input_resolution)
        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)
        # self.conv_block = DoubleConv(self.conv_dim, self.conv_dim, use_norm=False)
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.ReLU(True),
            nn.Conv2d(self.conv_dim, self.conv_dim, 3, 1, 1, padding_mode='reflect', bias=False)
        )
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
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        trans_x = self.trans_block(trans_x)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res

        return x


class SwinUp(nn.Module):
    def __init__(self, in_ch, out_ch, head_dim, window_size, dpr, dpr_flag, config, input_resolution, skip_ch=4):
        super(SwinUp, self).__init__()

        # self.dpr_flag=dpr_flag
        self.head_dim = head_dim
        # self.input_resolution = input_resolution

        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False)
        # self.skip = nn.Conv2d(out_ch, skip_ch, 2, 1, 0, bias=False)
        self.conv1 = nn.Conv2d(2*out_ch, out_ch, 3, 1, 1, padding_mode='reflect', bias=False)
        self.contact = Concat()
        self.process = [SwinConvBlock(out_ch // 2, out_ch // 2, self.head_dim, window_size, dpr[i + dpr_flag],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config)]
        self.process = nn.Sequential(*self.process)
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

    def forward(self, x, y):
        x = self.up(x)
        # y = self.skip(y)
        score = self.conv1(self.contact(x, y))
        # score = self.conv1(torch.cat((x, y), dim=1))
        score = self.process(score)
        return score


class SCUNet(nn.Module):

    def __init__(self, in_nc=1, config=[2, 2, 2, 2, 2, 2, 2], dim=64, head_dim=32, drop_path_rate=0.0,
                 input_resolution=128):
        super(SCUNet, self).__init__()
        self.config = config
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = 8

        # drop path rate for each layer
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]

        # self.m_head = [nn.Conv2d(in_nc * 2, dim, 3, 1, 1, padding_mode='reflect', bias=False)]
        # self.m_head = SingleConv(in_nc * 2, dim, use_norm=False)
        self.m_head =  MultiFeature(in_nc, 8, dim)
        begin = 0
        self.m_down1 = [SwinConvBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution)
                        for i in range(config[0])] + \
                       [nn.Conv2d(dim, 2 * dim, 2, 2, 0, bias=False)]

        begin += config[0]
        self.m_down2 = [SwinConvBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 2)
                        for i in range(config[1])] + \
                       [nn.Conv2d(2 * dim, 4 * dim, 2, 2, 0, bias=False)]

        begin += config[1]
        self.m_down3 = [SwinConvBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                       'W' if not i % 2 else 'SW', input_resolution // 4)
                        for i in range(config[2])] + \
                       [nn.Conv2d(4 * dim, 8 * dim, 2, 2, 0, bias=False)]

        begin += config[2]
        self.m_body = [SwinConvBlock(4 * dim, 4 * dim, self.head_dim, self.window_size, dpr[i + begin],
                                      'W' if not i % 2 else 'SW', input_resolution // 8)
                       for i in range(config[3])]

        begin += config[3]
        # self.m_up3 = [nn.ConvTranspose2d(8 * dim, 4 * dim, 2, 2, 0, bias=False), ] + \
        #               [ConvTransBlock(2 * dim, 2 * dim, self.head_dim, self.window_size, dpr[i + begin],
        #                               'W' if not i % 2 else 'SW', input_resolution // 4)
        #               for i in range(config[4])]
        self.up3 = SwinUp(8 * dim, 4 * dim, self.head_dim, self.window_size, dpr, begin, config[4],
                          input_resolution // 4)

        begin += config[4]
        # self.m_up2 = [nn.ConvTranspose2d(4 * dim, 2 * dim, 2, 2, 0, bias=False), ] + \
        #              [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i + begin],
        #                              'W' if not i % 2 else 'SW', input_resolution // 2)
        #               for i in range(config[5])]
        self.up2 = SwinUp(4 * dim, 2 * dim, self.head_dim, self.window_size, dpr, begin, config[5],
                          input_resolution // 2)

        begin += config[5]
        # self.m_up1 = [nn.ConvTranspose2d(2 * dim, dim, 2, 2, 0, bias=False), ] + \
        #              [ConvTransBlock(dim // 2, dim // 2, self.head_dim, self.window_size, dpr[i + begin],
        #                              'W' if not i % 2 else 'SW', input_resolution)
        #               for i in range(config[6])]
        self.up1 = SwinUp(2 * dim, dim, self.head_dim, self.window_size, dpr, begin, config[6], input_resolution)

        self.m_tail = [nn.Conv2d(dim, in_nc, 3, 1, 1, bias=False)]

        # self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        # self.m_up3 = nn.Sequential(*self.m_up3)
        # self.m_up2 = nn.Sequential(*self.m_up2)
        # self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)
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

    def forward(self, x0):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h / 64) * 64 - h)  #找到最接近64整数倍的数 向上
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.m_tail(x)
        # x = self.m_up3(x + x4)
        # x = self.m_up2(x + x3)
        # x = self.m_up1(x + x2)
        # x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        return x


if __name__ == '__main__':
    import time
    from NetModel.Net2D.block2d import Concat

    # config = [2, 2, 2, 2, 2, 2, 2]
    # dpr = [x.item() for x in torch.linspace(0, 0., sum(config))]
    # torch.cuda.empty_cache()
    # skip = nn.Conv2d(128, 4, 2, 1, bias=False)
    net = SCUNet(in_nc=1, dim=16, head_dim=16, input_resolution=128)
    # (self,in_ch,out_ch,head_dim,window_size,dpr,dpr_flag,config,input_resolution,skip_ch=4):
    # netup = SwinUp(32, 16, 16, 8, dpr, 4, 2, 256, skip_ch=4)
    # # x = torch.randn((3, 1, 64, 128))
    # up = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
    # x = torch.randn(10, 256, 8, 8)
    # y = torch.randn(10, 128, 16, 16)

    x1 = torch.randn(10, 1, 256, 256)
    # y1 = torch.randn(10, 1, 130, 130)
    # cat = Concat()
    # tog = nn.Conv2d(132, 128, 3, 1, 1, padding_mode='reflect', bias=False)
    # st_time = time.time()

    out = net(x1)
    # out1 = up(x)
    # out2 = skip(y)
    # out3 = cat(out1, out2)
    # out4 = tog(out3)
    # process = [ConvTransBlock(32 // 2, 32// 2,16, 8, dpr[i],
    #                                   'W' if not i % 2 else 'SW', 256)
    #                    for i in range(2)] 
    # process = nn.Sequential(*process)
    # out = net(x, y)
    # out1=process(y)
    # ed_time = time.time()
    # print(f"代码块运行时间: {ed_time - st_time:.4f} 秒")
    # dim=64
    # head_dim=32
    # window_size=8
    # drop_path_rate=0.
    # config=[2,2,2,2,2,2,2]
    # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
    # m_head = [nn.Conv2d(1, dim, 3, 1, 1, bias=False)]
    # m_down1 = [ConvTransBlock(dim//2, dim//2, head_dim, window_size, dpr[0], 'W' if not i%2 else 'SW', input_resolution=256) 
    #               for i in range(1)] + \
    #               [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]
    # m_body = [ConvTransBlock(2*dim, 2*dim, head_dim, window_size, dpr[0], 'W' if not i%2 else 'SW', 256//8)
    #         for i in range(config[3])]
    # m_up1 = [nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + \
    #                 [ConvTransBlock(dim//2, dim//2, head_dim, window_size, dpr[0], 'W' if not i%2 else 'SW', input_resolution=256) 
    #                   for i in range(config[6])]

    # m_head=nn.Sequential(*m_head)              
    # m_down1=nn.Sequential(*m_down1)
    # m_body=nn.Sequential(*m_body)
    # inc=m_head(x)
    # down1=m_down1(inc)
    # # x=m_body(down1)

    # print(m_head)
    # print(m_down1)
