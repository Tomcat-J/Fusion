# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from projects.mae_lite.utils import LayerNorm, GRN
from thop import profile
from timm.models import register_model


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=3,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    '''
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    '''

    def forward(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        f0 = x
        x = self.downsample_layers[1](x)
        x = self.stages[1](x)
        f1 = x
        x = self.downsample_layers[2](x)
        x = self.stages[2](x)
        f2 = x
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        f3 = x
        avg = self.norm(x.mean([-2, -1]))
        out = self.head(avg)

        feats = {}
        feats["feats"] = [f0, f1, f2, f3]
        feats["pooled_feat"] = avg
        return out, feats


@register_model
def convnextv2_atto(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs, drop_path_rate=0.2)
    return model


@register_model
def convnextv2_femto(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs, drop_path_rate=0.2)
    return model


@register_model
def convnextv2_pico(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs, drop_path_rate=0.2)
    return model


@register_model
def convnextv2_nano(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs, drop_path_rate=0.2)
    return model


@register_model
def convnextv2_tiny(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs, drop_path_rate=0.2)
    return model


@register_model
def convnextv2_base(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs, drop_path_rate=0.2)
    return model


@register_model
def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model


@register_model
def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


if __name__ == "__main__":
    import torch

    x = torch.randn(20, 3, 256, 256)
    net = convnextv2_tiny(num_classes=3)
    logit, feats = net(x)

    for f in feats["feats"]:
        print(f.shape, f.min().item())
    print(logit.shape)
    print(feats["pooled_feat"].shape)

    # model = convnextv2_tiny(num_classes=5)
    # input = torch.randn(1, 3, 256, 256)
    # flops, params = profile(model, inputs=(input,))
    #
    # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    # print("params=", str(params / 1e6) + '{}'.format("M"))
