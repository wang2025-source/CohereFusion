"""Shared image encoders, feature experts and frozen semantic priors."""


from collections import OrderedDict
import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class FrozenTextEncoder(nn.Module):
    """CLIP sentence and token features, without retaining the unused visual tower."""

    def __init__(self, model):
        super().__init__()
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, token_ids):
        dtype = self.text_projection.dtype
        x = self.token_embedding(token_ids).to(dtype)
        x = x + self.positional_embedding.to(dtype)
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x).float()
        rows = torch.arange(x.shape[0], device=x.device)
        sentence = x[rows, token_ids.argmax(-1)] @ self.text_projection.float()
        return F.normalize(x, dim=-1), F.normalize(sentence, dim=-1), token_ids.ne(0)


def rgb_to_ycbcr(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = .299 * r + .587 * g + .114 * b
    cb = .5 + .564 * (b - y)
    cr = .5 + .713 * (r - y)
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    cb, cr = cb - .5, cr - .5
    return torch.cat((y + 1.403 * cr,
                      y - .344 * cb - .714 * cr,
                      y + 1.773 * cb), 1)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class ImageEncoder(nn.Module):
    def __init__(self, inp_channels=3, dim=32, num_blocks=[2, 3, 3, 4], heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias'):
        super(ImageEncoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img_A):
        inp_enc_level1_A = self.patch_embed(inp_img_A)
        out_enc_level1_A = self.encoder_level1(inp_enc_level1_A)

        inp_enc_level2_A = self.down1_2(out_enc_level1_A)
        out_enc_level2_A = self.encoder_level2(inp_enc_level2_A)

        inp_enc_level3_A = self.down2_3(out_enc_level2_A)
        out_enc_level3_A = self.encoder_level3(inp_enc_level3_A)

        inp_enc_level4_A = self.down3_4(out_enc_level3_A)
        out_enc_level4_A = self.encoder_level4(inp_enc_level4_A)

        return out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A


class FrozenSemanticTeacher(nn.Module):
    """Use the public COCO/VOC DeepLabV3 teacher without distillation."""

    def __init__(self):
        super().__init__()
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights, deeplabv3_resnet50,
        )
        teacher = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        self.backbone = teacher.backbone
        self.backbone.requires_grad_(False).eval()
        self.register_buffer("mean", torch.tensor((.485, .456, .406))[None, :, None, None])
        self.register_buffer("std", torch.tensor((.229, .224, .225))[None, :, None, None])

    def train(self, mode=True):
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, visible, infrared):
        # Analyze both modalities in one teacher pass; IR is represented as RGB.
        size = (max(64, min(256, visible.shape[-2])),
                max(64, min(256, visible.shape[-1])))
        v = F.interpolate(visible.float(), size, mode="bilinear", align_corners=False)
        y, _, _ = rgb_to_ycbcr(infrared.float())
        i = F.interpolate(y.repeat(1, 3, 1, 1), size, mode="bilinear", align_corners=False)
        features = self.backbone((torch.cat((v, i), 0) - self.mean) / self.std)
        out = features["out"] if isinstance(features, (dict, OrderedDict)) else features
        return out.chunk(2, 0)


class SobelFeature(nn.Module):
    def __init__(self, channels):
        super().__init__()
        kx = torch.tensor(((-1., 0., 1.), (-2., 0., 2.), (-1., 0., 1.)))
        self.register_buffer("kx", kx[None, None].repeat(channels, 1, 1, 1))
        self.register_buffer("ky", kx.t()[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x):
        return (F.conv2d(x, self.kx, padding=1, groups=self.channels).abs() +
                F.conv2d(x, self.ky, padding=1, groups=self.channels).abs())


class Expert(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.GroupNorm(math.gcd(16, channels), channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)
