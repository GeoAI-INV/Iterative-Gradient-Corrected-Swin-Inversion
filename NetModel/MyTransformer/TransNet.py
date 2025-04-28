"""
Created on Mon Jun 17 17:02:43 2024
@author: Pang Qi
Notice: The code about 'SwinTransformerBlock' is rewritten from the original implementation available at:
        https://github.com/JingyunLiang/SwinIR
Intro:  'STMNet_1' - USTNet 
        'STMNet_1_all' - USTNet with any inputsize
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from NetModel.Net2D.block2d import SingleConv, DoubleConv, Concat, Regress
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward, DWTInverse


class PatchEmbedding(nn.Module):
    "(B,C,H,W)->(B,C*P*P,H/P,W/P)->(B,C*P*P,H*W/P^2)  dim:映射的层的数量"

    def __init__(self, in_channels, patch_size, dropout=0.):
        super().__init__()
        self.initialized = False
        self.image_size = 0
        image_height, image_width = to_2tuple(self.image_size)
        patch_height, patch_width = to_2tuple(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  #H*W/P^2

        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, patch_dim))
        self.dropout = nn.Dropout(dropout)

    def initialize(self, x):
        self.initialized = True
        self.image_size = x.shape[-1]

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  ## 加入cls 每一个token都对应一个
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        return x


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
    def __init__(self, in_ch, hidden_dim, out_dim, kernel_size=5, dilation=[1, 3, 6]):
        super().__init__()
        self.layers = len(dilation)
        self.muti_f = nn.ModuleList([SingleFeature(in_ch, hidden_dim, kernel_size, dilation[i])
                                     for i in range(self.layers)])
        self.concat = Concat()
        self.conv = DoubleConv(hidden_dim * self.layers, out_dim, use_norm=True)

    def forward(self, x):
        out_f = []
        for get_feature in self.muti_f:
            out_f.append(get_feature(x))
        scores = self.concat(*out_f)
        return self.conv(scores)


def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4 * (dilation_rate - 1),
                   stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_, 1, 1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2
        return flops


class WindowAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window 相对位置编码
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0] * self.win_size[1]
        nW = H * W / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H * W, H * W)

        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size = k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel * self.channel * self.k_size

        return flops


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x = self.linear1(x)
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        x = self.dwconv(x)
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)
        x = self.eca(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_mlp='leff',
                 modulator=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size * win_size, dim)  # modulator
        else:
            self.modulator = None

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn', 'mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == 'leff':
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask for SW-MSA 这里因为移位后有些相邻的位置是不需要进行attention的 因为他们的关系非常远
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
            # print('yes')

        shortcut = x
        # shortcut = self.simple_attention(x)
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows

        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


class SVT_channel_mixing(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.complex_weight_ll = nn.Parameter(torch.randn(dim, 32, 32, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
        self.complex_weight_lh_b2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink = 0.0

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        W = int(math.sqrt(N))
        x = x.view(B, H, W, C)
        x = torch.permute(x, (0, 3, 1, 2))
        B, C, H, W = x.shape
        x = x.to(torch.float32)

        xl, xh = self.xfm(x)
        xl = xl * self.complex_weight_ll

        xh[0] = torch.permute(xh[0], (5, 0, 2, 3, 4, 1))
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4],
                              self.num_blocks, self.block_size)

        x_real = xh[0][0]
        x_imag = xh[0][1]

        x_real_1 = F.relu(
            self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) +
            self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(
            self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) +
            self.complex_weight_lh_b1[1])

        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1,
                                                                                        self.complex_weight_lh_2[1]) + \
                   self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1,
                                                                                        self.complex_weight_lh_2[0]) + \
                   self.complex_weight_lh_b2[1]

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6])
        xh[0] = torch.permute(xh[0], (0, 4, 1, 2, 3, 5))

        x = self.ifm((xl, xh))
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(B, N, C)  # permute is not same as reshape or view
        return x


class ScaBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path=0., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = SVT_channel_mixing(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ScaEncoder(nn.Module):
    def __init__(self, dim, depth, mlp_ratio=4, drop_path=0., drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.blocks = nn.ModuleList([
            ScaBlock(dim=dim,
                     mlp_ratio=mlp_ratio,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, )
            for i in range(depth)])

    def forward(self, x):
        b, l, c = x.shape
        hh = int(math.sqrt(l))
        for blk in self.blocks:
            x = blk(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)
        return x


class STEncoder(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, depth, window_size=8, mlp_ratio=4, drop_path=0.,
                 use_norm=False):
        super().__init__()
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 win_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 # shift_size=0 if (i == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, )
            for i in range(depth)])

    def forward(self, x):
        # reg = x
        b, l, c = x.shape
        hh = int(math.sqrt(l))
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)
        return x


class STEncoder_sim(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, depth, window_size=8, mlp_ratio=4, drop_path=0.,
                 use_norm=False):
        super().__init__()
        self.input_resolution = input_resolution

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 win_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, )
            for i in range(depth)])
        self.simam = nn.Sequential(
            SingleConv(dim, dim, 3, use_norm),
            simam_module(),
            # se_module(dim),
            SingleConv(dim, dim, 1, use_norm)
        )

    def simple_attention(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = rearrange(x, 'b (h w) c -> b c h w ', h=H, w=W)  # view和 rearrange问题
        x = self.simam(x)
        x = rearrange(x, 'b c h w  -> b (h w) c')
        return x

    def forward(self, x):
        # reg = x
        b, l, c = x.shape
        x = [x]
        hh = int(math.sqrt(l))
        for i, blk in enumerate(self.blocks):
            x.append(blk(x[i]))
            # x = blk(x)
            if not i % 2 == 0:
                # print(i)
                x_f = self.simple_attention(x[i - 2])
                x[i + 1] = x_f + x[i + 1]
        x = rearrange(x[-1], 'b (h w) c -> b c h w', h=hh, w=hh)

        return x


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=False):
        super().__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.pool = nn.MaxPool2d(2)
        if use_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=to_pad),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=to_pad),
                nn.BatchNorm2d(out_ch))
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding=to_pad),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=to_pad))

    def forward(self, x, prompt_1=None):
        x = self.pool(x)
        x = self.double_conv(x)
        if prompt_1 is not None:
            prompt_1 = self.pool(prompt_1)
            prompt_1 = self.double_conv(prompt_1)
            return x * self.att(prompt_1), prompt_1
        else:
            return x, prompt_1


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, use_norm=False):
        super().__init__()
        to_pad = int((kernel_size - 1) / 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if use_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch + out_ch, out_ch, kernel_size, padding=to_pad),
                nn.BatchNorm2d(out_ch),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=to_pad),
                nn.BatchNorm2d(out_ch))
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_ch + out_ch, out_ch, kernel_size, padding=to_pad),
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=to_pad))

    def forward(self, x, x_pool):
        x = self.up(x)
        diff_h = x.size(2) - x_pool.size(2)
        diff_w = x.size(3) - x_pool.size(3)
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [-diff_w // 2, -diff_w + diff_w // 2,
                          -diff_h // 2, -diff_h + diff_h // 2])
        x = self.double_conv(torch.cat([x, x_pool], dim=1))
        return x


class OutProject(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1):
        super().__init__()
        to_pad = int((kernel_size - 1) / 2)  # make sure the same size between input and output
        self.activation = nn.Tanh()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=to_pad),
            # self.activation
        )

    def forward(self, x):
        return self.out_conv(x)


class WTFD(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.xfm = DWTForward(J=1, wave='haar', mode='zero')
        self.ifm = DWTInverse(wave='haar', mode='zero')
        self.compress_l = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.compress_h = nn.Conv2d(in_ch * 3, in_ch, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        x = self.compress_l(x)
        xl, xh = self.xfm(x)
        for i in range(len(xh)):
            xh[i] = xh[i].reshape(xh[i].shape[0], -1, xh[i].shape[-2], xh[i].shape[-1])
        lf = self.compress_l(xl)
        hf = self.compress_h(xh[0])
        hf = self.bn(hf)
        return lf, hf


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


class se_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def get_module_name():
        return "se"

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MFF(nn.Module):
    def __init__(self, embed_dim, input_size, heads, depth, window_size, drop_path):
        super().__init__()
        self.global_feature = STEncoder(embed_dim, [input_size, input_size], heads, depth=depth,
                                        window_size=window_size, drop_path=drop_path)
        self.lowhigh_feature = WTFD(embed_dim)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.compress = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, ),
            nn.BatchNorm2d(embed_dim))

    def forward(self, x):
        gl = self.global_feature(x.flatten(2).transpose(2, 1).contiguous())
        lf, hf = self.lowhigh_feature(x)
        lf = self.up(lf)
        fusion_feature = torch.cat((gl, lf), dim=1)
        return self.compress(fusion_feature)


class STMNet_1(nn.Module):
    def __init__(self, in_ch, out_ch, input_size, embed_dim, drop_path_rate=0.1, win_size=8, heads=1, dp=4,
                 use_norm=False):
        super().__init__()
        depths = [dp, dp, dp]
        self.heads = heads
        self.dropout = nn.Dropout(p=0.5)
        self.attn = simam_module()
        self.in_p = SingleConv(in_ch, embed_dim, use_norm=use_norm)

        down_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.st1 = STEncoder(embed_dim, [input_size, input_size], heads, depth=depths[0], window_size=win_size,
                             drop_path=down_dpr[sum(depths[:0]):sum(depths[:1])])
        self.down1 = Downsample(embed_dim, embed_dim * 2, use_norm=use_norm)
        self.st2 = STEncoder(embed_dim * 2, [input_size // 2, input_size // 2], heads * 2, depth=depths[1],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:1]):sum(depths[:2])])
        self.down2 = Downsample(embed_dim * 2, embed_dim * 4, use_norm=use_norm)
        self.st3 = STEncoder(embed_dim * 4, [input_size // 4, input_size // 4], heads * 4, depth=depths[2],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:2]):sum(depths[:3])])
        self.down3 = Downsample(embed_dim * 4, embed_dim * 8, use_norm=use_norm)

        bottle_dpr = [drop_path_rate] * 2
        self.bottle = STEncoder(embed_dim * 8, [input_size // 8, input_size // 8], heads * 8, depth=2,
                                window_size=win_size,
                                drop_path=bottle_dpr)

        self.up1 = Upsample(embed_dim * 8, embed_dim * 4)
        self.up2 = Upsample(embed_dim * 4, embed_dim * 2)
        self.up3 = Upsample(embed_dim * 2, embed_dim)
        self.out_p = Regress(embed_dim, out_ch)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, x_prompt=None):
        # x = torch.cat([x, prompt_1], dim=1)
        x_down = self.in_p(x)
        # x_down = self.dropout(x_down)
        x_st1 = self.st1(x_down.flatten(2).transpose(2, 1).contiguous())
        x_pool1, x_prompt1 = self.down1(x_st1, x_prompt)
        x_st2 = self.st2(x_pool1.flatten(2).transpose(2, 1).contiguous())
        x_pool2, x_prompt2 = self.down2(x_st2, x_prompt1)
        # print(self.st3)
        x_st3 = self.st3(x_pool2.flatten(2).transpose(2, 1).contiguous())
        x_pool3, x_prompt3 = self.down3(x_st3, x_prompt2)

        x_bottle = self.bottle(x_pool3.flatten(2).transpose(2, 1).contiguous())

        x_up1 = self.up1(x_bottle, x_st3)
        x_up2 = self.up2(x_up1, x_st2)
        x_up3 = self.up3(x_up2, x_st1)

        scores = self.out_p(x_up3)
        return scores


class STMNet_1_all(nn.Module):
    def __init__(self, in_ch, out_ch, input_size, embed_dim, drop_path_rate=0.1, win_size=8, heads=1, dp=4,
                 use_norm=False):
        super().__init__()
        depths = [dp, dp, dp]
        self.heads = heads
        self.dropout = nn.Dropout(p=0.5)
        self.attn = simam_module()
        self.in_p = SingleConv(in_ch, embed_dim, use_norm=use_norm)
        # self.in_s = SingleConv(in_ch, embed_dim, use_norm=use_norm)

        down_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.st1 = STEncoder(embed_dim, [input_size, input_size], heads, depth=depths[0], window_size=win_size,
                             drop_path=down_dpr[sum(depths[:0]):sum(depths[:1])])
        self.down1 = Downsample(embed_dim, embed_dim * 2, use_norm=use_norm)
        self.st2 = STEncoder(embed_dim * 2, [input_size // 2, input_size // 2], heads * 2, depth=depths[1],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:1]):sum(depths[:2])])
        self.down2 = Downsample(embed_dim * 2, embed_dim * 4, use_norm=use_norm)
        self.st3 = STEncoder(embed_dim * 4, [input_size // 4, input_size // 4], heads * 4, depth=depths[2],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:2]):sum(depths[:3])])
        self.down3 = Downsample(embed_dim * 4, embed_dim * 8, use_norm=use_norm)

        bottle_dpr = [drop_path_rate] * 2
        self.bottle = STEncoder(embed_dim * 8, [input_size // 8, input_size // 8], heads * 8, depth=2,
                                window_size=win_size,
                                drop_path=bottle_dpr)

        self.up1 = Upsample(embed_dim * 8, embed_dim * 4)
        self.up2 = Upsample(embed_dim * 4, embed_dim * 2)
        self.up3 = Upsample(embed_dim * 2, embed_dim)
        self.out_p = Regress(embed_dim, out_ch)
        # self.out_p = OutProject(embed_dim, out_ch)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, x_prompt=None):
        st_outputs = []
        pool_outputs = []
        up_outputs = []
        x_down = self.in_p(x)
        x_st1 = self.st1(x_down.flatten(2).transpose(2, 1).contiguous())
        x_pool1, x_prompt1 = self.down1(x_st1, x_prompt)
        st_outputs.append(x_st1)
        pool_outputs.append(x_pool1)

        x_st2 = self.st2(x_pool1.flatten(2).transpose(2, 1).contiguous())
        x_pool2, x_prompt2 = self.down2(x_st2, x_prompt1)
        st_outputs.append(x_st2)
        pool_outputs.append(x_pool2)

        x_st3 = self.st3(x_pool2.flatten(2).transpose(2, 1).contiguous())
        x_pool3, x_prompt3 = self.down3(x_st3, x_prompt2)
        st_outputs.append(x_st3)
        pool_outputs.append(x_pool3)

        x_bottle = self.bottle(x_pool3.flatten(2).transpose(2, 1).contiguous())
        st_outputs.append(x_bottle)

        x_up1 = self.up1(x_bottle, x_st3)
        up_outputs.append(x_up1)
        x_up2 = self.up2(x_up1, x_st2)
        up_outputs.append(x_up2)
        x_up3 = self.up3(x_up2, x_st1)
        up_outputs.append(x_up3)

        scores = self.out_p(x_up3)
        return scores, st_outputs, pool_outputs, up_outputs


class AESTMNet_1(nn.Module):
    def __init__(self, in_ch, out_ch, input_size, embed_dim, drop_path_rate=0.1, win_size=8,
                 use_norm=False, use_gpu=True):
        super().__init__()
        # if use_gpu:
        #     # self.ae = torch.load('/home/shendi_gjh_pq/workspace/IMP2d/checkpoints_pretrained/Jul02_211354_pretrained')
        #     self.ae = torch.load('D:\py_workspace\WorkSpace\AboutIMP\checkpoints_pretrained/Jul02_211354_pretrained')
        # else:
        #     # self.ae = torch.load('/home/shendi_gjh_pq/workspace/IMP2d/checkpoints_pretrained/Jul02_211354_pretrained').cpu()
        #     self.ae = torch.load(
        #         'D:\py_workspace\WorkSpace\AboutIMP\checkpoints_pretrained/Jul02_211354_pretrained').cpu()

        # self.xfm = DWTForward(J=1, wave='haar', mode='zero')
        # self.ifm = DWTInverse(wave='haar', mode='zero')
        self.in_p = SingleConv(in_ch, embed_dim, use_norm=use_norm)
        self.sima = simam_module()
        depths = [2, 2, 2]
        heads = 1

        down_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.st1 = STEncoder(embed_dim, [input_size, input_size], heads, depth=depths[0], window_size=win_size,
                             drop_path=down_dpr[sum(depths[:0]):sum(depths[:1])])
        self.down1 = Downsample(embed_dim, embed_dim * 2, use_norm=use_norm)
        # self.wt1 = WTFD(embed_dim)

        self.st2 = STEncoder(embed_dim * 2, [input_size // 2, input_size // 2], heads * 2, depth=depths[1],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:1]):sum(depths[:2])])
        self.down2 = Downsample(embed_dim * 2, embed_dim * 4, use_norm=use_norm)

        self.st3 = STEncoder(embed_dim * 4, [input_size // 4, input_size // 4], heads * 4, depth=depths[2],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:2]):sum(depths[:3])])
        self.down3 = Downsample(embed_dim * 4, embed_dim * 8, use_norm=use_norm)

        bottle_dpr = [drop_path_rate] * 2
        self.bottle = STEncoder(embed_dim * 8, [input_size // 8, input_size // 8], heads * 8, depth=2,
                                window_size=win_size,
                                drop_path=bottle_dpr)

        self.up1 = Upsample(embed_dim * 8, embed_dim * 4)
        self.up2 = Upsample(embed_dim * 4, embed_dim * 2)
        self.up3 = Upsample(embed_dim * 2, embed_dim)

        # self.out_p = Regress(embed_dim, out_ch)
        self.out_p = OutProject(embed_dim, out_ch)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x_down = self.sima(self.in_p(x))

        x_st1 = self.st1(x_down.flatten(2).transpose(2, 1).contiguous())
        x_pool1 = self.down1(self.sima(x_st1))
        # x_lf1 = self.wt1(x_down)[0]
        # x_pool1 = torch.cat((x_pool1,x_lf1), dim=1)

        x_st2 = self.st2(x_pool1.flatten(2).transpose(2, 1).contiguous())
        x_pool2 = self.down2(self.sima(x_st2))

        x_st3 = self.st3(x_pool2.flatten(2).transpose(2, 1).contiguous())
        x_pool3 = self.down3(self.sima(x_st3))

        x_bottle = self.bottle(x_pool3.flatten(2).transpose(2, 1).contiguous())

        x_up1 = self.up1(self.sima(x_bottle), x_st3)
        x_up2 = self.up2(self.sima(x_up1), x_st2)
        x_up3 = self.up3(self.sima(x_up2), x_st1)

        scores = self.out_p(x_up3)
        return scores


def pad_to_multiple(x, multiple):
    """
    对输入进行动态补零，使其大小为指定倍数的整数倍。
    Args:
        x (torch.Tensor): 输入张量，形状为 (B, C, H, W)。
        multiple (int): 希望补零后的大小为 multiple 的整数倍。
    Returns:
        padded_x (torch.Tensor): 补零后的张量。
    """
    H, W = x.shape[-2], x.shape[-1]
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    padded_x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
    return padded_x


class STMNet_3d(nn.Module):
    def __init__(self, in_ch, out_ch, input_size, embed_dim, drop_path_rate=0.1, win_size=8, heads=1, dp=4,
                 use_norm=False):
        super().__init__()
        depths = [dp, dp, dp]
        self.win_size = win_size
        self.heads = heads
        self.dropout = nn.Dropout(p=0.5)
        self.in_p = SingleConv(in_ch, embed_dim, use_norm=use_norm)
        # self.in_s = SingleConv(in_ch, embed_dim, use_norm=use_norm)

        down_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.st1 = STEncoder(embed_dim, [input_size, input_size], heads, depth=depths[0], window_size=win_size,
                             drop_path=down_dpr[sum(depths[:0]):sum(depths[:1])])
        self.down1 = Downsample(embed_dim, embed_dim * 2, use_norm=use_norm)
        self.st2 = STEncoder(embed_dim * 2, [input_size // 2, input_size // 2], heads * 2, depth=depths[1],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:1]):sum(depths[:2])])
        self.down2 = Downsample(embed_dim * 2, embed_dim * 4, use_norm=use_norm)
        self.st3 = STEncoder(embed_dim * 4, [input_size // 4, input_size // 4], heads * 4, depth=depths[2],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:2]):sum(depths[:3])])
        self.down3 = Downsample(embed_dim * 4, embed_dim * 8, use_norm=use_norm)

        bottle_dpr = [drop_path_rate] * 2
        self.bottle = STEncoder(embed_dim * 8, [input_size // 8, input_size // 8], heads * 8, depth=2,
                                window_size=win_size,
                                drop_path=bottle_dpr)

        self.up1 = Upsample(embed_dim * 8, embed_dim * 4)
        self.up2 = Upsample(embed_dim * 4, embed_dim * 2)
        self.up3 = Upsample(embed_dim * 2, embed_dim)
        self.out_p = Regress(embed_dim, out_ch)
        # self.out_p = OutProject(embed_dim, out_ch)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, x_prompt=None):
        # x = torch.cat([x, prompt_1], dim=1)
        x_down = self.in_p(x)
        # x_down = self.dropout(x_down)
        x_st1 = self.st1(x_down.flatten(2).transpose(2, 1).contiguous())
        x_pool1, x_prompt1 = self.down1(x_st1, x_prompt)
        x_pool1 = pad_to_multiple(x_pool1, self.win_size)

        x_st2 = self.st2(x_pool1.flatten(2).transpose(2, 1).contiguous())
        x_pool2, x_prompt2 = self.down2(x_st2, x_prompt1)
        x_pool2 = pad_to_multiple(x_pool2, self.win_size)

        x_st3 = self.st3(x_pool2.flatten(2).transpose(2, 1).contiguous())
        x_pool3, x_prompt3 = self.down3(x_st3, x_prompt2)
        x_pool3 = pad_to_multiple(x_pool3, self.win_size)
        x_bottle = self.bottle(x_pool3.flatten(2).transpose(2, 1).contiguous())

        x_up1 = self.up1(x_bottle, x_st3)
        x_up2 = self.up2(x_up1, x_st2)
        x_up3 = self.up3(x_up2, x_st1)
        scores = self.out_p(x_up3)
        return scores


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = x.permute(0, -1, -2, 1)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.permute(0, 2, 1).view(B, 2 * C, H // 2, W // 2)
        return x


class STMNet_patch(nn.Module):
    def __init__(self, in_ch, out_ch, input_size, embed_dim, drop_path_rate=0.1, win_size=8, heads=1, dp=4,
                 use_norm=False):
        super().__init__()
        depths = [dp, dp, dp]
        self.heads = heads
        self.dropout = nn.Dropout(p=0.5)
        self.in_p = SingleConv(in_ch, embed_dim, use_norm=use_norm)
        # self.in_s = SingleConv(in_ch, embed_dim, use_norm=use_norm)

        down_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.st1 = STEncoder(embed_dim, [input_size, input_size], heads, depth=depths[0], window_size=win_size,
                             drop_path=down_dpr[sum(depths[:0]):sum(depths[:1])])
        self.down1 = PatchMerging(dim=embed_dim)
        self.st2 = STEncoder(embed_dim * 2, [input_size // 2, input_size // 2], heads * 2, depth=depths[1],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:1]):sum(depths[:2])])
        self.down2 = PatchMerging(dim=embed_dim * 2)
        self.st3 = STEncoder(embed_dim * 4, [input_size // 4, input_size // 4], heads * 4, depth=depths[2],
                             window_size=win_size,
                             drop_path=down_dpr[sum(depths[:2]):sum(depths[:3])])
        self.down3 = PatchMerging(dim=embed_dim * 4)

        bottle_dpr = [drop_path_rate] * 2
        self.bottle = STEncoder(embed_dim * 8, [input_size // 8, input_size // 8], heads * 8, depth=2,
                                window_size=win_size,
                                drop_path=bottle_dpr)

        self.up1 = Upsample(embed_dim * 8, embed_dim * 4)
        self.up2 = Upsample(embed_dim * 4, embed_dim * 2)
        self.up3 = Upsample(embed_dim * 2, embed_dim)
        self.out_p = Regress(embed_dim, out_ch)
        # self.out_p = OutProject(embed_dim, out_ch)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, x_prompt=None):
        # x = torch.cat([x, prompt_1], dim=1)
        x_down = self.in_p(x)
        # x_down = self.dropout(x_down)
        x_st1 = self.st1(x_down.flatten(2).transpose(2, 1).contiguous())
        x_pool1 = self.down1(x_st1)
        x_st2 = self.st2(x_pool1.flatten(2).transpose(2, 1).contiguous())
        x_pool2 = self.down2(x_st2)
        # print(self.st3)
        x_st3 = self.st3(x_pool2.flatten(2).transpose(2, 1).contiguous())
        x_pool3 = self.down3(x_st3)

        x_bottle = self.bottle(x_pool3.flatten(2).transpose(2, 1).contiguous())

        x_up1 = self.up1(x_bottle, x_st3)
        x_up2 = self.up2(x_up1, x_st2)
        x_up3 = self.up3(x_up2, x_st1)

        scores = self.out_p(x_up3)
        return scores


if __name__ == '__main__':
    import time

    x = torch.randn(4, 2, 128, 128)
    net = STMNet_1(in_ch=2, out_ch=1, input_size=x.shape[2], embed_dim=8, drop_path_rate=0.1,
                       win_size=8, dp=2)
    out = net(x)
    print(out.shape)
