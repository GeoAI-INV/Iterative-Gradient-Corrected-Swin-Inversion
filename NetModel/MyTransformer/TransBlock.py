"""
Created on Sat Jun 15 09:39:03 2024
@author: Pang Qi
"""
import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PatchEmbedding(nn.Module):
    "(B,C,H,W)->(B,C*P*P,H/P,W/P)->(B,C*P*P,H*W/P^2)  dim:映射的层的数量"

    def __init__(self, in_channels, image_size, patch_size, dropout=0.):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)  # H*W/P^2

        patch_dim = in_channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, patch_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, patch_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        dim_head = dim // heads
        project_out = not (heads == 1)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        # self.apply(self._init_weights)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # Q*(K.T)/scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, depth, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dropout=dropout),
                FeedForward(dim, mlp_ratio, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x) + x
        return self.norm(x)


class Transformer_v2(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, depth, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dropout=dropout),
                FeedForward(dim, mlp_ratio, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, mlp in self.layers:
            x = attn(x) + x
            x = mlp(x)
        return x


class transencoder(nn.Module):
    def __init__(self, in_channels, image_size, patch_size, heads, depth, mlp_ratio=4, dropout=0., drop_path=0.,
                 use_cls=True
                 ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, image_size, patch_size)
        patch_dim = in_channels * patch_size ** 2
        patch_num = (image_size // patch_size) ** 2 + 1
        self.transformer = Transformer_v2(patch_dim, heads, mlp_ratio, depth, dropout)
        self.head = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(patch_num, patch_num - 1) if use_cls else nn.Identity(),
        )
        self.patch_size = patch_size
        self.image_size = image_size

    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.head(x)
        x = rearrange(x, 'b (c1 p1 p2) (k1 k2) -> b c1 (p1 k1) (p2 k2)', c1=c, p1=self.patch_size, k1=self.image_size // self.patch_size)
        x = x + shortcut
        return x


class ViT(nn.Module):
    def __init__(self, in_channels, image_size, patch_size, heads, depth, mlp_ratio=4, dropout=0., drop_path=0.,
                 use_cls=True
                 ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, image_size, patch_size)
        patch_dim = in_channels * patch_size ** 2
        patch_num = (image_size // patch_size) ** 2 + 1
        self.transformer = Transformer(patch_dim, heads, mlp_ratio, depth, dropout)
        self.head = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(patch_num, patch_num - 1) if use_cls else nn.Identity(),
        )
        self.patch_size = patch_size
        self.image_size = image_size

    def forward(self, x):
        shortcut = x
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.head(x)
        x = rearrange(x, 'b (c1 p1 p2) (k1 k2) -> b c1 (p1 k1) (p2 k2)', c1=c, p1=self.patch_size, k1=self.image_size // self.patch_size)
        x = x + shortcut
        return x


if __name__ == "__main__":
    x = torch.randn(10, 1, 224, 224)
    y = torch.randn(10, 1, 224, 224)
    net = transencoder(in_channels=1, image_size=224, patch_size=14, heads=1, depth=1)
    # net2 = PatchEmbedding(in_channels=1,  image_size=224, patch_size=14)
    # net3 = Attention(dim=1*14*14, heads=4)
    out = net(x)
    # out2 = net2(x)
    # out3 = net3(out2)
