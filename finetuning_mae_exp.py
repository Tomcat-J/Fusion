# # --------------------------------------------------------
# # Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# # --------------------------------------------------------
# # Modified from MAE (https://github.com/facebookresearch/mae)
# # Copyright (c) Meta Platforms, Inc. and affiliates.
# # All rights reserved.
# # --------------------------------------------------------
# # References:
# # timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# # DeiT: https://github.com/facebookresearch/deit
# # --------------------------------------------------------
from functools import partial
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
# from mmengine.model import BaseModule, normal_init
from timm.models.vision_transformer import PatchEmbed
from mae_lite.models.models_vit import Block
from projects.mae_lite.util.pos_embed import get_2d_sincos_pos_embed
from timm.models import register_model

# MHF_block_v2 可从 mhf_enhancement.py 导入用于增强版模型
# from projects.mae_lite.mhf_enhancement import MHF_block_v2
#
# # class LayerNorm(nn.Module):
# #     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
# #     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
# #     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
# #     with shape (batch_size, channels, height, width).
# #     """
# #
# #     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
# #         super().__init__()
# #         self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
# #         self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
# #         self.eps = eps
# #         self.data_format = data_format
# #         if self.data_format not in ["channels_last", "channels_first"]:
# #             raise ValueError(f"not support data format '{self.data_format}'")
# #         self.normalized_shape = (normalized_shape,)
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         if self.data_format == "channels_last":
# #             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
# #         elif self.data_format == "channels_first":
# #             # [batch_size, channels, height, width]
# #             mean = x.mean(1, keepdim=True)
# #             var = (x - mean).pow(2).mean(1, keepdim=True)
# #             x = (x - mean) / torch.sqrt(var + self.eps)
# #             x = self.weight[:, None, None] * x + self.bias[:, None, None]
# #             return x
# #
# # def drop_path_f(x, drop_prob: float = 0., training: bool = False):
# #     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
# #
# #     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
# #     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
# #     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
# #     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
# #     'survival rate' as the argument.
# #
# #     """
# #     if drop_prob == 0. or not training:
# #         return x
# #     keep_prob = 1 - drop_prob
# #     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
# #     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
# #     random_tensor.floor_()  # binarize
# #     output = x.div(keep_prob) * random_tensor
# #     return output
# #
# #
# # class DropPath(nn.Module):
# #     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
# #     """
# #     def __init__(self, drop_prob=None):
# #         super(DropPath, self).__init__()
# #         self.drop_prob = drop_prob
# #
# #     def forward(self, x):
# #         return drop_path_f(x, self.drop_prob, self.training)
# # class Local_block(nn.Module):
# #     r""" Local Feature Block. There are two equivalent implementations:
# #     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
# #     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
# #     We use (2) as we find it slightly faster in PyTorch
# #
# #     Args:
# #         dim (int): Number of input channels.
# #         drop_rate (float): Stochastic depth rate. Default: 0.0
# #     """
# #     def __init__(self, dim, drop_rate=0.):
# #         super().__init__()
# #         self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
# #         self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
# #         self.pwconv = nn.Linear(dim, dim)  # pointwise/1x1 convs, implemented with linear layers
# #         self.act = nn.GELU()
# #         self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
# #
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         shortcut = x
# #         x = self.dwconv(x)
# #         x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
# #         x = self.norm(x)
# #         x = self.pwconv(x)
# #         x = self.act(x)
# #         x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
# #         x = shortcut + self.drop_path(x)
# #         return x
# #
# # class Conv(nn.Module):
# #     def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
# #         super(Conv, self).__init__()
# #         self.inp_dim = inp_dim
# #         self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
# #         self.relu = None
# #         self.bn = None
# #         if relu:
# #             self.relu = nn.ReLU(inplace=True)
# #         if bn:
# #             self.bn = nn.BatchNorm2d(out_dim)
# #
# #     def forward(self, x):
# #         assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
# #         x = self.conv(x)
# #         if self.bn is not None:
# #             x = self.bn(x)
# #         if self.relu is not None:
# #             x = self.relu(x)
# #         return x
# #
# # #### Inverted Residual MLP
# # class IRMLP(nn.Module):
# #     def __init__(self, inp_dim, out_dim):
# #         super(IRMLP, self).__init__()
# #         self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
# #         self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
# #         self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
# #         self.gelu = nn.GELU()
# #         self.bn1 = nn.BatchNorm2d(inp_dim)
# #
# #     def forward(self, x):
# #
# #         residual = x
# #         out = self.conv1(x)
# #         out = self.gelu(out)
# #         out += residual
# #
# #         out = self.bn1(out)
# #         out = self.conv2(out)
# #         out = self.gelu(out)
# #         out = self.conv3(out)
# #
# #         return out
# #
# # class HFF_block(nn.Module):
# #     def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
# #         super(HFF_block, self).__init__()
# #         self.maxpool=nn.AdaptiveMaxPool2d(1)
# #         self.avgpool=nn.AdaptiveAvgPool2d(1)
# #         self.se=nn.Sequential(
# #             nn.Conv2d(ch_2, ch_2 // r_2, 1,bias=False),
# #             nn.ReLU(),
# #             nn.Conv2d(ch_2 // r_2, ch_2, 1,bias=False)
# #         )
# #         self.sigmoid = nn.Sigmoid()
# #         self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
# #         self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
# #         self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
# #         self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
# #         self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
# #         self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
# #         self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)
# #
# #         self.gelu = nn.GELU()
# #
# #         self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
# #         self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
# #
# #     def forward(self, l, g):
# #         # 检查 l 和 g 的空间尺寸是否相同
# #         if l.size(2) != g.size(2) or l.size(3) != g.size(3):
# #             # 调整 l 的尺寸以匹配 g
# #             l = F.interpolate(l, size=(g.size(2), g.size(3)), mode='nearest')
# #
# #         W_local = self.W_l(l)   # local feature from Local Feature Block
# #         W_global = self.W_g(g)   # global feature from Global Feature Block
# #         shortcut = 0
# #         X_f = torch.cat([W_local, W_global], 1)
# #         X_f = self.norm2(X_f)
# #         X_f = self.W(X_f)
# #         X_f = self.gelu(X_f)
# #
# #         # spatial attention for ConvNeXt branch
# #         l_jump = l
# #         max_result, _ = torch.max(l, dim=1, keepdim=True)
# #         avg_result = torch.mean(l, dim=1, keepdim=True)
# #         result = torch.cat([max_result, avg_result], 1)
# #         l = self.spatial(result)
# #         l = self.sigmoid(l) * l_jump
# #
# #         # channel attetion for transformer branch
# #         g_jump = g
# #         max_result=self.maxpool(g)
# #         avg_result=self.avgpool(g)
# #         max_out=self.se(max_result)
# #         avg_out=self.se(avg_result)
# #         g = self.sigmoid(max_out+avg_out) * g_jump
# #
# #         fuse = torch.cat([g, l, X_f], 1)
# #         fuse = self.norm3(fuse)
# #         fuse = self.residual(fuse)
# #         fuse = shortcut + self.drop_path(fuse)
# #         return fuse
#
# class MaskedAutoencoderViT(nn.Module):
#     """Masked Autoencoder with VisionTransformer backbone"""
#
#     def __init__(
#         self,
#         num_classes=3,
#         img_size=224,
#         patch_size=16,
#         in_chans=3,
#         embed_dim=1024,
#         depth=12,
#         num_heads=16,
#         decoder_embed_dim=512,
#         decoder_pred_dim=None,
#         decoder_depth=8,
#         decoder_num_heads=16,
#         mlp_ratio=4.0,
#         out_indices=[1, 3, 9, 11],
#         norm_layer=nn.LayerNorm,
#         norm_pix_loss=False,
#         # distilled = True,
#         # HFF_dp=0.,
#         # conv_depths=(2, 2, 2, 2), conv_dims=(96, 192, 384, 768), conv_drop_path_rate=0.,
#         # conv_head_init_scale: float = 1.
#
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.depth = depth
#         self.num_heads = num_heads
#         self.out_indices = out_indices
#         # self.distilled = distilled
#         #----------------------
#         # if self.distilled:
#         #     self.downsample_layers = nn.ModuleList()   # stem + 3 stage downsample
#         #     stem = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),
#         #                          LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
#         #     self.downsample_layers.append(stem)
#         #
#         #     # stage2-4 downsample
#         #     for i in range(3):
#         #         downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
#         #                                          nn.Conv2d(conv_dims[i], conv_dims[i+1], kernel_size=2, stride=2))
#         #         self.downsample_layers.append(downsample_layer)
#         #     self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
#         #     dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))]
#         #     cur = 0
#         #
#         #     # Build stacks of blocks in each stage
#         #     for i in range(4):
#         #         stage = nn.Sequential(
#         #             *[Local_block(dim=conv_dims[i], drop_rate=dp_rates[cur + j])
#         #               for j in range(conv_depths[i])]
#         #         )
#         #         self.stages.append((stage))
#         #         cur += conv_depths[i]
#         #
#         #     self.conv_norm = nn.LayerNorm(conv_dims[-1], eps=1e-6)   # final norm layer
#         #     self.conv_head = nn.Linear(conv_dims[-1], num_classes)
#         #     self.conv_head.weight.data.mul_(conv_head_init_scale)
#         #     self.conv_head.bias.data.mul_(conv_head_init_scale)
#
#         #----------------------
#         # MAE encoder specifics
#         self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
#         num_patches = self.patch_embed.num_patches
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
#         )  # fixed sin-cos embedding
#
#         self.blocks = nn.ModuleList(
#             [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for i in range(depth)]
#         )
#         self.norm = norm_layer(embed_dim)
#         # --------------------------------------------------------------------------
#
#         # --------------------------------------------------------------------------
#         # MAE decoder specifics
#         self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
#
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
#
#         self.decoder_pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
#         )  # fixed sin-cos embedding
#
#         self.decoder_blocks = nn.ModuleList(
#             [
#                 Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
#                 for i in range(decoder_depth)
#             ]
#         )
#         self.decoder_embed_dim = decoder_embed_dim
#         self.decoder_norm = norm_layer(decoder_embed_dim)
#         decoder_pred_dim = patch_size ** 2 * in_chans if decoder_pred_dim is None else decoder_pred_dim
#         self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_pred_dim, bias=True)  # encoder to decoder
#         # --------------------------------------------------------------------------
#         ###### Hierachical Feature Fusion Block Setting #######
#         #
#         # self.fu1 = HFF_block(ch_1=96, ch_2=384, r_2=16, ch_int=256, ch_out=384, drop_rate=HFF_dp)
#         # self.fu2 = HFF_block(ch_1=192, ch_2=384, r_2=16, ch_int=256, ch_out=384, drop_rate=HFF_dp)
#         # self.fu3 = HFF_block(ch_1=384, ch_2=384, r_2=16, ch_int=384, ch_out=384, drop_rate=HFF_dp)
#         # self.fu4 = HFF_block(ch_1=768, ch_2=384, r_2=16, ch_int=512, ch_out=384, drop_rate=HFF_dp)
#
#
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.fc_norm = norm_layer(embed_dim)
#         self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
#         self.norm_pix_loss = norm_pix_loss
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         # initialization
#         # initialize (and freeze) pos_embed by sin-cos embedding
#         pos_embed = get_2d_sincos_pos_embed(
#             self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True
#         )
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#         decoder_pos_embed = get_2d_sincos_pos_embed(
#             self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True
#         )
#         self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
#
#         # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#
#         # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#         torch.nn.init.normal_(self.cls_token, std=0.02)
#         torch.nn.init.normal_(self.mask_token, std=0.02)
#
#         # initialize nn.Linear and nn.LayerNorm
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
#             nn.init.trunc_normal_(m.weight, std=0.2)
#             nn.init.constant_(m.bias, 0)
#     def patchify(self, imgs):
#         """
#         imgs: (N, 3, H, W)
#         x: (N, L, patch_size**2 *3)
#         """
#         p = self.patch_embed.patch_size[0]
#         assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
#
#         h = w = imgs.shape[2] // p
#         x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
#         x = torch.einsum("nchpwq->nhwpqc", x)
#         x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
#         return x
#
#     def unpatchify(self, x):
#         """
#         x: (N, L, patch_size**2 *3)
#         imgs: (N, 3, H, W)
#         """
#         p = self.patch_embed.patch_size[0]
#         h = w = int(x.shape[1] ** 0.5)
#         assert h * w == x.shape[1]
#
#         x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
#         x = torch.einsum("nhwpqc->nchpwq", x)
#         imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
#         return imgs
#
#     def random_masking(self, x, mask_ratio, ids_shuffle=None):
#         """
#         Perform per-sample random masking by per-sample shuffling.
#         Per-sample shuffling is done by argsort random noise.
#         x: [N, L, D], sequence
#         """
#         N, L, D = x.shape  # batch, length, dim
#         len_keep = int(L * (1 - mask_ratio))
#         if ids_shuffle is None:
#             noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
#             # sort noise for each sample
#             ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
#
#         ids_restore = torch.argsort(ids_shuffle, dim=1)
#
#         # keep the first subset
#         ids_keep = ids_shuffle[:, :len_keep]
#         x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
#
#         # generate the binary mask: 0 is keep, 1 is remove
#         mask = torch.ones([N, L], device=x.device)
#         mask[:, :len_keep] = 0
#         # unshuffle to get the binary mask
#         mask = torch.gather(mask, dim=1, index=ids_restore)
#
#         return x_masked, mask, ids_restore, ids_shuffle
#
#     def forward_encoder(self, x, mask_ratio, ids_shuffle=None):
#         # embed patches
#         x = self.patch_embed(x)
#
#         # add pos embed w/o cls token
#         x = x + self.pos_embed[:, 1:, :]
#
#         # masking: length -> length * mask_ratio
#         x, mask, ids_restore, ids_shuffle = self.random_masking(x, mask_ratio, ids_shuffle)
#
#         # append cls token
#         cls_token = self.cls_token + self.pos_embed[:, :1, :]
#         cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#
#         # apply Transformer blocks
#         for blk in self.blocks:
#             outcome = blk(x)
#
#         outcome = self.norm(outcome)
#
#         return outcome, mask, ids_restore, ids_shuffle
#
#     def forward_decoder(self, x, ids_restore):
#         # embed tokens
#         x = self.decoder_embed(x)
#
#         # append mask tokens to sequence
#         mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
#         x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
#         x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
#         x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
#
#         # add pos embed
#         x = x + self.decoder_pos_embed
#
#         # apply Transformer blocks
#         for blk in self.decoder_blocks:
#             x = blk(x)
#         x = self.decoder_norm(x)
#
#         # predictor projection
#         x = self.decoder_pred(x)
#
#         # remove cls token
#         x = x[:, 1:, :]
#
#         return x
#
#     def forward_features(self, x):
#         # if self.distilled:
#         #     ######  Local Branch ######
#         #     x_c = self.downsample_layers[0](x)
#         #     x_c_1 = self.stages[0](x_c)
#         #     x_c = self.downsample_layers[1](x_c_1)
#         #     x_c_2 = self.stages[1](x_c)
#         #     x_c = self.downsample_layers[2](x_c_2)
#         #     x_c_3 = self.stages[2](x_c)
#         #     x_c = self.downsample_layers[3](x_c_3)
#         #     x_c_4 = self.stages[3](x_c)
#         #
#         #     x_c = {}
#         #     x_c[1] = x_c_1
#         #     x_c[3] = x_c_2
#         #     x_c[9] = x_c_3
#         #     x_c[11] = x_c_4
#
#         x = self.patch_embed(x)
#         B, N, C = x.shape
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = x + self.pos_embed
#         feats = {}
#         feats["feats"] = []
#         # fu = []
#         # fu.append(self.fu1)
#         # fu.append(self.fu2)
#         # fu.append(self.fu3)
#         # fu.append(self.fu4)
#
#         for i,blk in enumerate(self.blocks):
#             x = blk(x)
#             if i in self.out_indices:
#                 x1 = x[:,1:,:].transpose(1, 2).reshape(B,-1,self.patch_embed.grid_size[0],self.patch_embed.grid_size[1])
#                 feats["feats"].append(x1)
#                 # if self.distilled:
#                 #     x = fu[i](x_c[i],x1).transpose(1,2).reshape(B,N,C)
#         avg = self.avgpool(x.transpose(1,2))
#         avg = torch.flatten(avg,1)
#         feats["pooled_feat"] = avg
#         # if self.distilled:
#         #     ###### Hierachical Feature Fusion Path ######
#         #     x_f_1 = self.fu1(x_c_1, feats["feats"][0], None)
#         #     x_f_2 = self.fu2(x_c_2, feats["feats"][1], x_f_1)
#         #     x_f_3 = self.fu3(x_c_3, feats["feats"][2], x_f_2)
#         #     x_f_4 = self.fu4(x_c_4, feats["feats"][3], x_f_3)
#         #
#         #     x = x_f_4.transpose(1,2).reshape(B,N,C)
#         # outcome = x[:, :1, :]
#         outcome = x[:, 0, :]
#         outcome = self.fc_norm(outcome)
#
#         return outcome,feats
#
#     def forward_loss(self, imgs, pred, mask):
#         """
#         imgs: [N, 3, H, W]
#         pred: [N, L, p*p*3]
#         mask: [N, L], 0 is keep, 1 is remove,
#         """
#         target = self.patchify(imgs)
#         if self.norm_pix_loss:
#             mean = target.mean(dim=-1, keepdim=True)
#             var = target.var(dim=-1, keepdim=True)
#             target = (target - mean) / (var + 1.0e-6) ** 0.5
#
#         loss = (pred - target) ** 2
#         loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
#
#         loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
#         return loss
#
#
#     def forward(self, imgs, mask_ratio=0.75, ids_shuffle=None):
#         if self.training:
#             latent, mask, ids_restore, ids_shuffle = self.forward_encoder(imgs, mask_ratio, ids_shuffle)
#             pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
#             loss = self.forward_loss(imgs, pred, mask)
#             return loss, pred, mask, ids_shuffle
#         else:
#             x,_ = self.forward_features(imgs)
#             x = self.head(x)
#             return x
#
#
# @register_model
# def mae_vit_tiny_patch16(pretrained=False, **kwargs):
#     # the number of heads is changed to 12 from 3, which is different to the original arch.
#     model = MaskedAutoencoderViT(
#         patch_size=16,
#         embed_dim=192,
#         depth=12,
#         num_heads=12,
#         decoder_embed_dim=96,
#         decoder_depth=1,
#         decoder_num_heads=3,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model
#
#
# @register_model
# def mae_vit_small_patch16(pretrained=False, **kwargs):
#     # the number of heads is changed to 12 from 6, which is different to the original arch.
#     model = MaskedAutoencoderViT(
#         patch_size=16,
#         embed_dim=384,
#         depth=12,
#         num_heads=12,
#         decoder_embed_dim=192,
#         decoder_depth=1,
#         decoder_num_heads=6,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model
#
#
# @register_model
# def mae_vit_base_patch16(pretrained=False, **kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16,
#         embed_dim=768,
#         depth=12,
#         num_heads=12,
#         decoder_embed_dim=512,
#         decoder_depth=8,
#         decoder_num_heads=16,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model
#
#
# @register_model
# def mae_vit_large_patch16(pretrained=False, **kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=16,
#         embed_dim=1024,
#         depth=24,
#         num_heads=16,
#         decoder_embed_dim=512,
#         decoder_depth=8,
#         decoder_num_heads=16,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model
#
#
# @register_model
# def mae_vit_huge_patch14(pretrained=False, **kwargs):
#     model = MaskedAutoencoderViT(
#         patch_size=14,
#         embed_dim=1280,
#         depth=32,
#         num_heads=16,
#         decoder_embed_dim=512,
#         decoder_depth=8,
#         decoder_num_heads=16,
#         mlp_ratio=4,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         **kwargs
#     )
#     return model
#
# if __name__ == "__main__":
#     import torch
#
#     x = torch.randn(32,3,256,256)
#     net = mae_vit_small_patch16()
#     logit, feats = net.forward_features(x)
#
#     for f in feats["feats"]:
#         print(f.shape, f.min().item())
#     print(logit.shape)
#     print(feats["pooled_feat"].shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

from timm.models import register_model
from timm.models.vision_transformer import Block as Blockmeta

# 使用优化版多原型分类头 (含软正交损失、局部一致性损失、CB加权)
from projects.mae_lite.head_optimal import HeadOptimal as Head, create_optimal_head_for_main_model


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


class main_model(nn.Module):

    def __init__(self, num_classes, patch_size=4, in_chans=3, embed_dim=96, depths=(1, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0,
                 attn_drop_rate=0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
                 conv_depths: list = [3, 3, 9, 3], conv_dims: list = [96, 192, 384, 768],
                 conv_drop_path_rate: float = 0.,
                 conv_layer_scale_init_value: float = 1e-6,
                 conv_head_init_scale: float = 1.,
                 **kwargs):
        super().__init__()

        ###### ConvNeXt Branch Setting #######

        self.downsample_layers = nn.ModuleList()  # stem + 3 stage downsample
        stem = nn.Sequential(nn.Conv2d(in_chans, conv_dims[0], kernel_size=4, stride=4),
                             LayerNorm(conv_dims[0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # stage2-4 downsample
        for i in range(3):
            downsample_layer = nn.Sequential(LayerNorm(conv_dims[i], eps=1e-6, data_format="channels_first"),
                                             # nn.Conv2d(conv_dims[i], 4*conv_dims[i], kernel_size=2, dilation=2, padding=1),
                                             nn.Conv2d(conv_dims[i], conv_dims[i + 1], kernel_size=2, stride=2))
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        dp_rates = [x.item() for x in torch.linspace(0, conv_drop_path_rate, sum(conv_depths))]
        cur = 0

        # Build stacks of blocks in each stage
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=conv_dims[i], drop_rate=dp_rates[cur + j],
                        layer_scale_init_value=conv_layer_scale_init_value)
                  for j in range(conv_depths[i])]
            )
            self.stages.append((stage))
            cur += conv_depths[i]

        self.conv_norm = nn.LayerNorm(768, eps=1e-6)  # final norm layer conv_dims[-1]
        # self.conv_head = nn.Linear(conv_dims[-1], num_classes)
        # self.conv_head.weight.data.mul_(conv_head_init_scale)
        # self.conv_head.bias.data.mul_(conv_head_init_scale)

        ###### Swin Transformer Branch Setting ######

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # The channels of stage4 output feature matrix
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.meta_tokenlize = PatchEmbed(patch_size=16, in_c=96, embed_dim=768)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        i_layer = 0
        self.layers1 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  mlp_ratio=self.mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 1
        self.layers2 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  mlp_ratio=self.mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 2
        self.layers3 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  mlp_ratio=self.mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        i_layer = 3
        self.layers4 = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                  depth=depths[i_layer],
                                  num_heads=num_heads[i_layer],
                                  window_size=window_size,
                                  mlp_ratio=self.mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  drop=drop_rate,
                                  attn_drop=attn_drop_rate,
                                  drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                  norm_layer=norm_layer,
                                  downsample=PatchMerging if (i_layer > 0) else None,
                                  use_checkpoint=use_checkpoint)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 使用优化版多原型分类头 (HeadOptimal)
        # 特性: 软正交损失、局部一致性损失、CB加权、高动量push
        self.head = Head(
            num_classes=self.num_classes,
            emb_dim=conv_dims[-1],
            num_heads=num_heads[2],
            img_feat_dim=conv_dims[-1],
            num_prototypes=5,           # 每类初始原型数
            max_prototypes=10,          # 每类最大原型数
            use_class_balanced=True,    # 启用 Class-Balanced 加权
            cb_beta=0.9999,             # CB 系数
            local_consistency_scale=0.05,  # 局部一致性损失权重
        ) if num_classes > 0 else nn.Identity()
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        ###### Hierachical Feature Fusion Block Setting #######

        self.fu1 = MHF_block(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96, drop_rate=0.)
        self.fu2 = MHF_block(ch_1=192, ch_2=192, r_2=16, ch_int=192, ch_out=192, drop_rate=0.1)
        self.fu3 = MHF_block(ch_1=384, ch_2=384, r_2=16, ch_int=384, ch_out=384, drop_rate=0.2)
        self.fu4 = MHF_block(ch_1=768, ch_2=768, r_2=16, ch_int=768, ch_out=768, drop_rate=0.2)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward_train(self, images):

        ######  Swin Transformer Branch ######
        x_s, H, W = self.patch_embed(images)
        x_s = self.pos_drop(x_s)
        x_s_1, H, W = self.layers1(x_s, H, W)

        # [B,L,C] ---> [B,C,H,W]
        x_s_1 = torch.transpose(x_s_1, 1, 2)
        x_s_1 = x_s_1.view(x_s_1.shape[0], -1, 56, 56)

        ######  ConvNeXt Branch ######
        x_c = self.downsample_layers[0](images)
        x_c_1 = self.stages[0](x_c)

        ###### Hierachical Feature Fusion Path ######
        x_f_1 = self.fu1(x_c_1, x_s_1, None)
        # x_f_1 = x_s_1 + x_c_1
        ###  [B,C,H,W] ---> [B,L,C]
        B, C, H, W = x_f_1.shape
        x_s_2 = x_f_1.reshape(B, -1, C)

        x_s_2, H, W = self.layers2(x_s_2, H, W)
        x_s_2 = torch.transpose(x_s_2, 1, 2)
        x_s_2 = x_s_2.view(x_s_2.shape[0], -1, 28, 28)
        x_c_2 = self.downsample_layers[1](x_f_1)
        x_c_2 = self.stages[1](x_c_2)
        x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1)
        # x_f_2 = x_s_2 + x_c_2
        B, C, H, W = x_f_2.shape
        x_s_3 = x_f_2.reshape(B, -1, C)
        x_s_3, H, W = self.layers3(x_s_3, H, W)
        x_s_3 = torch.transpose(x_s_3, 1, 2)
        x_s_3 = x_s_3.view(x_s_3.shape[0], -1, 14, 14)
        x_c_3 = self.downsample_layers[2](x_f_2)
        x_c_3 = self.stages[2](x_c_3)
        x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
        # x_f_3 = x_s_3 + x_c_3
        B, C, H, W = x_f_3.shape
        x_s_4 = x_f_3.reshape(B, -1, C)
        x_s_4, H, W = self.layers4(x_s_4, H, W)
        x_s_4 = torch.transpose(x_s_4, 1, 2)
        x_s_4 = x_s_4.view(x_s_4.shape[0], -1, 7, 7)
        x_c_4 = self.downsample_layers[3](x_f_3)
        x_c_4 = self.stages[3](x_c_4)
        x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3)
        # x_f_4 = x_s_4 + x_c_4
        B, C, _, _ = x_f_4.shape

        x_all = x_f_4.mean([-2, -1]).unsqueeze(1)  # global average pooling (N, C, H, W) -> (N, C)
        x_fu = self.conv_norm(x_all)
        x_fu1 = self.conv_norm(x_f_4.reshape(B, C, -1).transpose(1,2))
        # x_fu = self.conv_norm(x_f_4.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)


        return x_fu,x_fu1

    # def forward(self,images):
    #     if self.training:
    #         features,poolling_feature = self.forward_train(images)
    #         logits,loss = self.head(poolling_feature,features)
    #         # logits = self.conv_head(features)
    #         return logits,loss
    #     else:
    #         features,poolling_feature = self.forward_train(images)
    #         logits = self.head(poolling_feature,features)
    #         return logits

    def forward(self, images, target=None):
        """
        前向传播
        Args:
            images: 输入图像 (B, C, H, W)
            target: 标签 (B,)，训练时需要
        Returns:
            训练时: (loss, extra_dict) - 用于 train.py
            评估时: (logits, head_loss) - 用于 run_eval
        """
        features, poolling_feature = self.forward_train(images)
        
        if self.training and target is not None:
            # 训练时: head 返回 (logits, total_loss, loss_dict) 或 (logits, loss)
            head_output = self.head(poolling_feature, features, labels=target)
            
            if len(head_output) == 3:
                # HeadOptimal 返回 (logits, total_loss, loss_dict)
                logits, total_loss, loss_dict = head_output
                extra_dict = {
                    'total_loss': total_loss.detach(),
                    **{k: v.detach() if isinstance(v, torch.Tensor) else v 
                       for k, v in loss_dict.items()}
                }
            else:
                # 兼容 Head 返回 (logits, loss)
                logits, total_loss = head_output
                extra_dict = {'total_loss': total_loss.detach()}
            
            return total_loss, extra_dict
        else:
            # 评估/推理时
            head_output = self.head(poolling_feature, features, labels=target)
            if len(head_output) == 3:
                logits, head_loss, _ = head_output
            else:
                logits, head_loss = head_output
            return logits, head_loss

    @torch.no_grad()
    def push_prototypes(self, dataloader, device, use_local=True, momentum=0.9):
        """
        原型推送方法 - 支持全局和局部两种模式
        
        Args:
            dataloader: 数据加载器，返回 (images, labels)
            device: GPU 设备
            use_local: 是否使用局部 patch 特征 (默认 True，推荐)
                - True: 使用 x_fu1 (B, 49, 768) 局部 patches
                - False: 使用 x_fu (B, 1, 768) 全局池化
            momentum: 动量系数 (默认 0.9)
        """
        if use_local:
            # 推荐: 使用局部 patch 特征
            return self.push_prototypes_local(dataloader, device, momentum)
        else:
            # 备选: 使用全局池化特征
            return self._push_prototypes_global(dataloader, device)
    
    @torch.no_grad()
    def _push_prototypes_global(self, dataloader, device):
        """
        原型推送方法 - 在主模型中实现，用于提取特征并更新原型
        
        Args:
            dataloader: 数据加载器，返回 (images, labels)
            device: GPU 设备
        """
        # 检查 head 是否支持 push
        if not hasattr(self.head, 'proto_manager'):
            print("Head does not support prototype push, skipping...")
            return
        
        self.eval()
        
        # 收集每个类别的特征
        class_features = [[] for _ in range(self.num_classes)]
        
        print("Collecting features for prototype push...")
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 提取特征 (不经过 head)
            features, pooling_features = self.forward_train(images)
            
            # 投影特征到 head 的嵌入空间
            proj_features = self.head.proj(features.squeeze(1))  # (B, emb_dim)
            
            # 按类别收集
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_features[c].append(proj_features[mask].detach().cpu())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
        
        print("Updating prototypes...")
        proto_manager = self.head.proto_manager
        var_threshold = proto_manager.var_threshold
        
        # 处理每个类别
        for c in range(self.num_classes):
            if not class_features[c]:
                continue
            
            all_feats = torch.cat(class_features[c], dim=0).to(device)  # (num_samples, emb_dim)
            
            # 1. 检查是否需要扩展原型 (基于方差)
            feat_var = all_feats.var(dim=0).mean().item()
            if feat_var > var_threshold:
                try:
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
                    kmeans.fit(all_feats.cpu().numpy())
                    new_proto = torch.tensor(kmeans.cluster_centers_[0], device=device, dtype=all_feats.dtype)
                    if proto_manager.expand_prototypes(c, new_proto):
                        print(f"  Class {c}: Expanded prototype (var={feat_var:.4f})")
                except ImportError:
                    pass
            
            # 2. 动量更新现有原型
            proto_indices = proto_manager.get_prototype_indices()
            start, end = proto_indices[c]
            class_protos = proto_manager.prototypes[start:end]  # (num_protos, emb_dim)
            
            # 计算每个原型与所有特征的距离
            dists = torch.cdist(class_protos, all_feats)  # (num_protos, num_samples)
            
            # 找到每个原型最近的特征
            min_dists, min_idx = dists.min(dim=1)
            
            for k in range(end - start):
                proto_idx = start + k
                closest_feat = all_feats[min_idx[k]]
                proto_manager.momentum_update(proto_idx, closest_feat)
        
        # 应用动量更新
        proto_manager.apply_momentum_updates()
        print("Prototype push completed!")

    @torch.no_grad()
    def push_prototypes_local(self, dataloader, device, momentum=0.9, max_patches_per_class=5000):
        """
        局部特征 Push - 将原型推送到最相似的局部 patch 特征
        
        与原 push_prototypes 的核心区别:
        - 原方法: 使用 x_fu (B, 1, 768) 全局池化特征
        - 新方法: 使用 x_fu1 (B, 49, 768) 局部 patch 特征
        
        理论依据:
        - ProtoPNet (NIPS 2019): 原型应锚定到局部 patch
        - ProtoASNet (CVPR 2023): 局部对齐优于全局池化
        - GPA (2025): Local alignment 显著提升原型多样性
        
        Args:
            dataloader: 数据加载器，返回 (images, labels)
            device: GPU 设备
            momentum: 动量系数 (默认 0.9，高动量防止漂移)
            max_patches_per_class: 每类最大收集的 patch 数 (防止内存溢出)
        """
        # 检查 head 是否支持 push
        if not hasattr(self.head, 'proto_manager'):
            print("Head does not support prototype push, skipping...")
            return
        
        self.eval()
        proto_manager = self.head.proto_manager
        emb_dim = proto_manager.emb_dim
        
        # 收集每个类别的局部 patch 特征
        class_patches = [[] for _ in range(self.num_classes)]
        class_patch_counts = [0] * self.num_classes
        
        print("Collecting LOCAL patch features for prototype push...")
        print(f"  Using x_fu1 (B, 49, 768) instead of x_fu (B, 1, 768)")
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 提取特征: x_fu (全局), x_fu1 (局部 patches)
            x_fu, x_fu1 = self.forward_train(images)
            # x_fu: (B, 1, 768) - 全局池化特征 (不使用)
            # x_fu1: (B, 49, 768) - 局部 patch 特征 (使用这个!)
            
            B, L, C = x_fu1.shape  # L = 49 (7x7 patches)
            
            # 投影到 head 嵌入空间
            local_feats = self.head.proj(x_fu1.reshape(B * L, C))  # (B*49, emb_dim)
            local_feats = local_feats.view(B, L, -1)  # (B, 49, emb_dim)
            
            # 按类别收集
            for c in range(self.num_classes):
                if class_patch_counts[c] >= max_patches_per_class:
                    continue  # 已收集足够的 patches
                    
                mask = (labels == c)
                if mask.sum() > 0:
                    # 每个样本有 L=49 个 patches
                    c_patches = local_feats[mask].reshape(-1, emb_dim)  # (N_c * 49, emb_dim)
                    
                    # 限制收集数量
                    remaining = max_patches_per_class - class_patch_counts[c]
                    if c_patches.size(0) > remaining:
                        c_patches = c_patches[:remaining]
                    
                    class_patches[c].append(c_patches.detach().cpu())
                    class_patch_counts[c] += c_patches.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1} batches...")
                # 检查是否所有类别都收集够了
                if all(cnt >= max_patches_per_class for cnt in class_patch_counts):
                    print(f"  All classes reached max patches ({max_patches_per_class}), stopping collection.")
                    break
        
        # 打印收集统计
        print("Patch collection statistics:")
        for c in range(self.num_classes):
            print(f"  Class {c}: {class_patch_counts[c]} patches")
        
        print("Updating prototypes using LOCAL patches...")
        
        # 处理每个类别
        for c in range(self.num_classes):
            if not class_patches[c]:
                print(f"  Class {c}: No patches collected, skipping...")
                continue
            
            # 合并所有 patches
            all_patches = torch.cat(class_patches[c], dim=0).to(device)  # (N_c * 49, emb_dim)
            print(f"  Class {c}: {all_patches.size(0)} patches")
            
            # 获取该类别的原型索引
            proto_indices = proto_manager.get_prototype_indices()
            start, end = proto_indices[c]
            num_protos = end - start
            
            if num_protos == 0:
                continue
            
            # 获取当前原型
            class_protos = proto_manager.prototypes[start:end].clone()  # (num_protos, emb_dim)
            
            # 计算每个原型与所有 patches 的距离
            # 使用 L2 距离
            dists = torch.cdist(class_protos, all_patches)  # (num_protos, num_patches)
            
            # 找到每个原型最近的 patch
            min_dists, min_idx = dists.min(dim=1)
            
            # 动量更新每个原型
            for k in range(num_protos):
                proto_idx = start + k
                nearest_patch = all_patches[min_idx[k]]
                
                # 动量更新: p = m * p + (1-m) * nearest
                old_proto = proto_manager.prototypes.data[proto_idx]
                new_proto = momentum * old_proto + (1 - momentum) * nearest_patch
                new_proto = F.normalize(new_proto, p=2, dim=-1)
                proto_manager.prototypes.data[proto_idx] = new_proto
                
            print(f"    Updated {num_protos} prototypes, min_dist: {min_dists.mean().item():.4f}")
        
        print("Local prototype push completed!")

    # def forward_train(self, imgs):

        # ######  Global Branch ######
        # x_s, H, W = self.patch_embed(imgs)
        # x_s = self.pos_drop(x_s)
        # x_s_1, H, W = self.layers1(x_s, H, W)
        # x_s_2, H, W = self.layers2(x_s_1, H, W)
        # x_s_3, H, W = self.layers3(x_s_2, H, W)
        # x_s_4, H, W = self.layers4(x_s_3, H, W)
        #
        # # [B,L,C] ---> [B,C,H,W]
        # x_s_1 = torch.transpose(x_s_1, 1, 2)
        # x_s_1 = x_s_1.view(x_s_1.shape[0], -1, 56, 56)
        # x_s_2 = torch.transpose(x_s_2, 1, 2)
        # x_s_2 = x_s_2.view(x_s_2.shape[0], -1, 28, 28)
        # x_s_3 = torch.transpose(x_s_3, 1, 2)
        # x_s_3 = x_s_3.view(x_s_3.shape[0], -1, 14, 14)
        # x_s_4 = torch.transpose(x_s_4, 1, 2)
        # x_s_4 = x_s_4.view(x_s_4.shape[0], -1, 7, 7)

        ######  Local Branch ######
        # x_c = self.downsample_layers[0](imgs)
        # x_c_1 = self.stages[0](x_c)
        # x_c = self.downsample_layers[1](x_c_1)
        # x_c_2 = self.stages[1](x_c)
        # x_c = self.downsample_layers[2](x_c_2)
        # x_c_3 = self.stages[2](x_c)
        # x_c = self.downsample_layers[3](x_c_3)
        # x_c_4 = self.stages[3](x_c)

        # ###### Hierachical Feature Fusion Path ######
        # x_f_1 = self.fu1(x_c_1, x_s_1, None)
        # x_f_2 = self.fu2(x_c_2, x_s_2, x_f_1)
        # x_f_3 = self.fu3(x_c_3, x_s_3, x_f_2)
        # x_f_4 = self.fu4(x_c_4, x_s_4, x_f_3)
        # B, C, _, _ = x_s_4.shape

        # x_all = x_c_4.mean([-2, -1]).unsqueeze(1)  # global average pooling (N, C, H, W) -> (N, C)
        # x_fu = self.conv_norm(x_all)
        # x_fu1 = self.conv_norm(x_s_4.reshape(B, C, -1).transpose(1,2))
        # x_fu = self.conv_norm(x_s_4.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        #
        # return x_fu,x_fu1


##### ConvNeXt Component #####

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_rate (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x


# Hierachical Feature Fusion Block
class MHF_block(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(MHF_block, self).__init__()
        #---------------
        self.att_conv = nn.Sequential(
                nn.Conv2d(ch_int * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        #---------------
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_x = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.norm1 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.chan = Conv(ch_int // 2, ch_int, 1, bn=True, relu=True)
        self.W = Conv(ch_int * 2, ch_int, 3, bn=True, relu=True)
        # self.W3 = Conv(ch_int * 3, ch_int, 3, bn=True, relu=True)
        # self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, l, g, f):
        n, _, h, w = g.shape
        W_local = self.W_x(l)  # local feature from ConvNeXt
        W_global = self.W_g(g)  # global feature from SwinTransformer
        # fusion
        z1 = torch.cat([W_global, W_local], dim=1)
        z1 = self.att_conv(z1)
        W = W_global * z1[:, 0].view(n, 1, h, w) + W_local * z1[:, 1].view(n, 1, h, w)
        if f is not None:
            W_f = self.chan(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W],1)
            X_f = self.norm1(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # spatial attention for ConvNeXt branch
        l_jump = l
        max_result, _ = torch.max(l, dim=1, keepdim=True)
        avg_result = torch.mean(l, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        l = self.spatial(result)
        l = self.sigmoid(l) * l_jump

        # channel attetion for transformer branch
        g_jump = g
        max_result = self.maxpool(g)
        avg_result = self.avgpool(g)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        g = self.sigmoid(output) * g_jump

        fuse = self.residual(torch.cat([g, l, X_f], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse) + shortcut
        else:
            return fuse + shortcut


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.double_conv(x) + self.identity(x))


# SwinTransformer
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size

        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):

        if self.downsample is not None:
            x = self.downsample(x, H, W)  # patch merging stage2 in [6,3136,96] out [6,784,192]
            H, W = (H + 1) // 2, (W + 1) // 2

        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:  # swin block
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)

        return x, H, W


def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # downsample patch_size times
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        dim = dim // 2
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


@register_model
def HiFuse_Tiny(num_classes: int, pretrained=False, **kwargs):
    model = main_model(depths=(2, 2, 2, 2),
                       conv_depths=(2, 2, 2, 2),
                       num_classes=num_classes,
                       **kwargs)
    return model


@register_model
def HiFuse_Small(num_classes: int, pretrained=False, **kwargs):
    model = main_model(depths=(2, 2, 6, 2),
                       conv_depths=(3, 3, 9, 3),
                       num_classes=num_classes,
                       **kwargs)
    return model


@register_model
def HiFuse_Base(num_classes: int, pretrained=False):
    model = main_model(depths=(2, 2, 18, 2),
                       conv_depths=(2, 2, 18, 2),
                       num_classes=num_classes)
    return model


# ============================================================================
# MHF Enhancement: 增强版模型 (独立类，不影响原始main_model)
# ============================================================================

class main_model_enhanced(main_model):
    """
    增强版main_model，使用MHF_block_v2替代MHF_block
    
    继承自main_model，只覆盖MHF_block相关部分，保持其他代码不变。
    这样可以：
    1. 保持原始main_model完全不变，方便跑baseline对比实验
    2. 增强版模型与原始模型权重兼容（fu1-fu4名称相同）
    3. 支持差异化学习率和冻结/解冻控制
    
    使用方法:
        from projects.mae_lite.models_mae import main_model_enhanced
        model = main_model_enhanced(num_classes=7, enhancement_reduction=0.0625)
    """
    
    def __init__(self, num_classes, patch_size=4, in_chans=3, embed_dim=96, depths=(1, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0,
                 attn_drop_rate=0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
                 conv_depths: list = [3, 3, 9, 3], conv_dims: list = [96, 192, 384, 768],
                 conv_drop_path_rate: float = 0.,
                 conv_layer_scale_init_value: float = 1e-6,
                 conv_head_init_scale: float = 1.,
                 enhancement_reduction: float = 0.0625,
                 **kwargs):
        # 调用父类初始化（会创建原始MHF_block）
        super().__init__(
            num_classes=num_classes, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            patch_norm=patch_norm, use_checkpoint=use_checkpoint,
            conv_depths=conv_depths, conv_dims=conv_dims,
            conv_drop_path_rate=conv_drop_path_rate,
            conv_layer_scale_init_value=conv_layer_scale_init_value,
            conv_head_init_scale=conv_head_init_scale,
            **kwargs
        )
        
        # 导入并替换为MHF_block_v2
        from projects.mae_lite.mhf_enhancement import MHF_block_v2
        
        self.enhancement_reduction = enhancement_reduction
        
        # 用MHF_block_v2替换原始MHF_block
        self.fu1 = MHF_block_v2(ch_1=96, ch_2=96, r_2=16, ch_int=96, ch_out=96, drop_rate=0.,
                                use_enhancement=True, enhancement_reduction=enhancement_reduction)
        self.fu2 = MHF_block_v2(ch_1=192, ch_2=192, r_2=16, ch_int=192, ch_out=192, drop_rate=0.1,
                                use_enhancement=True, enhancement_reduction=enhancement_reduction)
        self.fu3 = MHF_block_v2(ch_1=384, ch_2=384, r_2=16, ch_int=384, ch_out=384, drop_rate=0.2,
                                use_enhancement=True, enhancement_reduction=enhancement_reduction)
        self.fu4 = MHF_block_v2(ch_1=768, ch_2=768, r_2=16, ch_int=768, ch_out=768, drop_rate=0.2,
                                use_enhancement=True, enhancement_reduction=enhancement_reduction)
        
        print(f"[main_model_enhanced] MHF Enhancement: Enabled (reduction={enhancement_reduction})")
    
    def freeze_backbone(self):
        """
        冻结backbone参数（ConvNeXt和Swin Transformer分支）
        """
        frozen_count = 0
        for name, param in self.named_parameters():
            if not any(x in name for x in ['fu1', 'fu2', 'fu3', 'fu4', 'head']):
                param.requires_grad = False
                frozen_count += 1
        print(f"[main_model_enhanced] Backbone frozen: {frozen_count} parameter tensors")
    
    def unfreeze_backbone(self):
        """
        解冻所有backbone参数
        """
        unfrozen_count = 0
        for param in self.parameters():
            if not param.requires_grad:
                param.requires_grad = True
                unfrozen_count += 1
        print(f"[main_model_enhanced] Backbone unfrozen: {unfrozen_count} parameter tensors")
    
    def freeze_mhf_original(self):
        """
        冻结MHF_block的原始参数（不包括enhancement模块）
        """
        frozen_count = 0
        for name, param in self.named_parameters():
            if any(x in name for x in ['fu1', 'fu2', 'fu3', 'fu4']):
                if 'enhancement' not in name:
                    param.requires_grad = False
                    frozen_count += 1
        print(f"[main_model_enhanced] MHF original frozen: {frozen_count} parameter tensors")


@register_model
def HiFuse_Tiny_Enhanced(num_classes: int, enhancement_reduction: float = 0.0625, pretrained=False):
    """增强版HiFuse_Tiny"""
    model = main_model_enhanced(depths=(1, 2, 6, 2),
                                conv_depths=(3, 3, 9, 3),
                                num_classes=num_classes,
                                enhancement_reduction=enhancement_reduction)
    return model


@register_model
def HiFuse_Small_Enhanced(num_classes: int, enhancement_reduction: float = 0.0625, pretrained=False, **kwargs):
    """增强版HiFuse_Small"""
    # kwargs 接受但忽略 drop_rate, drop_path_rate, attn_drop_rate, drop_block_rate, global_pool 等参数
    model = main_model_enhanced(depths=(2, 2, 6, 2),
                                conv_depths=(3, 3, 9, 3),
                                num_classes=num_classes,
                                enhancement_reduction=enhancement_reduction)
    return model


@register_model
def HiFuse_Base_Enhanced(num_classes: int, enhancement_reduction: float = 0.0625, pretrained=False, **kwargs):
    """增强版HiFuse_Base"""
    # kwargs 接受但忽略 drop_rate, drop_path_rate, attn_drop_rate, drop_block_rate, global_pool 等参数
    model = main_model_enhanced(depths=(2, 2, 18, 2),
                                conv_depths=(2, 2, 18, 2),
                                num_classes=num_classes,
                                enhancement_reduction=enhancement_reduction)
    return model


# """
# original code from facebook research:
# https://github.com/facebookresearch/ConvNeXt
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
#
# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs
#     with shape (batch_size, channels, height, width).
#     """
#
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
#         self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
#         self.eps = eps
#         self.data_format = data_format
#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise ValueError(f"not support data format '{self.data_format}'")
#         self.normalized_shape = (normalized_shape,)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         elif self.data_format == "channels_first":
#             # [batch_size, channels, height, width]
#             mean = x.mean(1, keepdim=True)
#             var = (x - mean).pow(2).mean(1, keepdim=True)
#             x = (x - mean) / torch.sqrt(var + self.eps)
#             x = self.weight[:, None, None] * x + self.bias[:, None, None]
#             return x
#
#
# class Block(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
#
#     Args:
#         dim (int): Number of input channels.
#         drop_rate (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#     def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
#         self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
#                                   requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         shortcut = x
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
#         x = self.norm(x)
#         x = self.pwconv1(x)
#         x = self.act(x)
#         x = self.pwconv2(x)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
#
#         x = shortcut + self.drop_path(x)
#         return x
#
#
# class ConvNeXt(nn.Module):
#     r""" ConvNeXt
#         A PyTorch impl of : `A ConvNet for the 2020s`  -
#           https://arxiv.org/pdf/2201.03545.pdf
#     Args:
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
#         dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#         head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
#     """
#     def __init__(self, in_chans: int = 3, num_classes: int = 3, depths: list = None,
#                  dims: list = None, drop_path_rate: float = 0., layer_scale_init_value: float = 1e-6,
#                  head_init_scale: float = 1.):
#         super().__init__()
#         self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
#         stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
#                              LayerNorm(dims[0], eps=1e-6, data_format="channels_first"))
#         self.downsample_layers.append(stem)
#
#         # 对应stage2-stage4前的3个downsample
#         for i in range(3):
#             downsample_layer = nn.Sequential(LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
#                                              nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2))
#             self.downsample_layers.append(downsample_layer)
#
#         self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
#         dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         cur = 0
#         # 构建每个stage中堆叠的block
#         for i in range(4):
#             stage = nn.Sequential(
#                 *[Block(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
#                   for j in range(depths[i])]
#             )
#             self.stages.append(stage)
#             cur += depths[i]
#
#         self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
#         self.head = nn.Linear(dims[-1], num_classes)
#         self.apply(self._init_weights)
#         self.head.weight.data.mul_(head_init_scale)
#         self.head.bias.data.mul_(head_init_scale)
#
#     def _init_weights(self, m):
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             nn.init.trunc_normal_(m.weight, std=0.2)
#             nn.init.constant_(m.bias, 0)
#
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#
#
#     def forward_features(self, x: torch.Tensor) -> torch.Tensor:
#         for i in range(4):
#             x = self.downsample_layers[i](x)
#             x = self.stages[i](x)
#
#         return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x
#
#
#     # def forward(self,x):
#     #     x = self.downsample_layers[0](x)
#     #     x = self.stages[0](x)
#     #     f0 = x
#     #     x = self.downsample_layers[1](x)
#     #     x = self.stages[1](x)
#     #     f1 = x
#     #     x = self.downsample_layers[2](x)
#     #     x = self.stages[2](x)
#     #     f2 = x
#     #     x = self.downsample_layers[3](x)
#     #     x = self.stages[3](x)
#     #     f3 = x
#     #     avg = self.norm(x.mean([-2, -1]))
#     #     out = self.head(avg)
#     #
#     #     feats = {}
#     #     feats["feats"] = [f0, f1, f2, f3]
#     #     feats["pooled_feat"] = avg
#     #     return out,feats
#
# @register_model
# def convnext_tiny(num_classes: int):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth
#     model = ConvNeXt(depths=[3, 3, 9, 3],
#                      dims=[96, 192, 384, 768],
#                      num_classes=num_classes)
#     return model
#
# @register_model
# def convnext_small(num_classes: int):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth
#     model = ConvNeXt(depths=[3, 3, 27, 3],
#                      dims=[96, 192, 384, 768],
#                      num_classes=num_classes)
#     return model
#
# @register_model
# def convnext_base(num_classes: int,pretrained=False, **kwargs):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
#     # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
#     model = ConvNeXt(depths=[3, 3, 27, 3],
#                      dims=[128, 256, 512, 1024],
#                      num_classes=num_classes,
#                      drop_path_rate=0.2)
#     return model
#
# @register_model
# def convnext_large(num_classes: int):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth
#     # https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth
#     model = ConvNeXt(depths=[3, 3, 27, 3],
#                      dims=[192, 384, 768, 1536],
#                      num_classes=num_classes)
#     return model
#
# @register_model
# def convnext_xlarge(num_classes: int):
#     # https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth
#     model = ConvNeXt(depths=[3, 3, 27, 3],
#                      dims=[256, 512, 1024, 2048],
#                      num_classes=num_classes)
#     return model

# __all__ = [
#     'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
#     'vgg19_bn', 'vgg19',
# ]
#
#
# model_urls = {
#     'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
#     'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
#     'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
#     'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
# }
#
#
# class VGG(nn.Module):
#
#     def __init__(self, features, num_classes=3):
#         super(VGG, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, num_classes),
#         )
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#
# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
#
#
# cfg = {
#     'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }
#
#
# def vgg11(pretrained=False, model_root=None, **kwargs):
#     """VGG 11-layer model (configuration "A")"""
#     model = VGG(make_layers(cfg['A']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
#     return model
#
#
# def vgg11_bn(**kwargs):
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     kwargs.pop('model_root', None)
#     return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
#
#
# def vgg13(pretrained=False, model_root=None, **kwargs):
#     """VGG 13-layer model (configuration "B")"""
#     model = VGG(make_layers(cfg['B']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
#     return model
#
#
# def vgg13_bn(**kwargs):
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     kwargs.pop('model_root', None)
#     return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
#
#
# def vgg16(pretrained=False, model_root=None, **kwargs):
#     """VGG 16-layer model (configuration "D")"""
#     model = VGG(make_layers(cfg['D']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
#     return model
#
#
# def vgg16_bn(**kwargs):
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     kwargs.pop('model_root', None)
#     return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
#
# def vgg19_bn(**kwargs):
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     kwargs.pop('model_root', None)
#     return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
#
# @register_model
# def vgg19(pretrained=False, model_root=None, **kwargs):
#     """VGG 19-layer model (configuration "E")"""
#     model = VGG(make_layers(cfg['E']), **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
#     return model







# """ Swin Transformer
# A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
#     - https://arxiv.org/pdf/2103.14030
#
# Code/weights from https://github.com/microsoft/Swin-Transformer
#
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
# import numpy as np
# from typing import Optional
#
#
# def drop_path_f(x, drop_prob: float = 0., training: bool = False):
#     """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# class DropPath(nn.Module):
#     """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path_f(x, self.drop_prob, self.training)
#
#
# def window_partition(x, window_size: int):
#     """
#     将feature map按照window_size划分成一个个没有重叠的window
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size(M)
#
#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
#     # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows
#
#
# def window_reverse(windows, window_size: int, H: int, W: int):
#     """
#     将一个个window还原成一个feature map
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size(M)
#         H (int): Height of image
#         W (int): Width of image
#
#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
#     # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x
#
#
# class PatchEmbed(nn.Module):
#     """
#     2D Image to Patch Embedding
#     """
#     def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
#         super().__init__()
#         patch_size = (patch_size, patch_size)
#         self.patch_size = patch_size
#         self.in_chans = in_c
#         self.embed_dim = embed_dim
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         _, _, H, W = x.shape
#
#         # padding
#         # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
#         pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
#         if pad_input:
#             # to pad the last 3 dimensions,
#             # (W_left, W_right, H_top,H_bottom, C_front, C_back)
#             x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
#                           0, self.patch_size[0] - H % self.patch_size[0],
#                           0, 0))
#
#         # 下采样patch_size倍
#         x = self.proj(x)
#         _, _, H, W = x.shape
#         # flatten: [B, C, H, W] -> [B, C, HW]
#         # transpose: [B, C, HW] -> [B, HW, C]
#         x = x.flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x, H, W
#
#
# class PatchMerging(nn.Module):
#     r""" Patch Merging Layer.
#
#     Args:
#         dim (int): Number of input channels.
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self, dim, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
#         self.norm = norm_layer(4 * dim)
#
#     def forward(self, x, H, W):
#         """
#         x: B, H*W, C
#         """
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#
#         x = x.view(B, H, W, C)
#
#         # padding
#         # 如果输入feature map的H，W不是2的整数倍，需要进行padding
#         pad_input = (H % 2 == 1) or (W % 2 == 1)
#         if pad_input:
#             # to pad the last 3 dimensions, starting from the last dimension and moving forward.
#             # (C_front, C_back, W_left, W_right, H_top, H_bottom)
#             # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
#             x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
#
#         x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
#         x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
#         x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
#         x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
#         x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
#         x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]
#
#         x = self.norm(x)
#         x = self.reduction(x)  # [B, H/2*W/2, 2*C]
#
#         return x
#
#
# class Mlp(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop2 = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x
#
#
# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.
#
#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """
#
#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
#
#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # [Mh, Mw]
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
#
#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
#         coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
#         # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
#         self.register_buffer("relative_position_index", relative_position_index)
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, mask: Optional[torch.Tensor] = None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, Mh*Mw, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         # [batch_size*num_windows, Mh*Mw, total_embed_dim]
#         B_, N, C = x.shape
#         # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
#         # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
#         # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
#         q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
#
#         # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
#         # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))
#
#         # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
#         attn = attn + relative_position_bias.unsqueeze(0)
#
#         if mask is not None:
#             # mask: [nW, Mh*Mw, Mh*Mw]
#             nW = mask.shape[0]  # num_windows
#             # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
#             # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)
#
#         attn = self.attn_drop(attn)
#
#         # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
#         # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class SwinTransformerBlock(nn.Module):
#     r""" Swin Transformer Block.
#
#     Args:
#         dim (int): Number of input channels.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#     """
#
#     def __init__(self, dim, num_heads, window_size=7, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
#
#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
#             attn_drop=attn_drop, proj_drop=drop)
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#
#     def forward(self, x, attn_mask):
#         H, W = self.H, self.W
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#
#         shortcut = x
#         x = self.norm1(x)
#         x = x.view(B, H, W, C)
#
#         # pad feature maps to multiples of window size
#         # 把feature map给pad到window size的整数倍
#         pad_l = pad_t = 0
#         pad_r = (self.window_size - W % self.window_size) % self.window_size
#         pad_b = (self.window_size - H % self.window_size) % self.window_size
#         x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
#         _, Hp, Wp, _ = x.shape
#
#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x
#             attn_mask = None
#
#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]
#
#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]
#
#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
#         shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]
#
#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x
#
#         if pad_r > 0 or pad_b > 0:
#             # 把前面pad的数据移除掉
#             x = x[:, :H, :W, :].contiguous()
#
#         x = x.view(B, H * W, C)
#
#         # FFN
#         x = shortcut + self.drop_path(x)
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#
#         return x
#
#
# class BasicLayer(nn.Module):
#     """
#     A basic Swin Transformer layer for one stage.
#
#     Args:
#         dim (int): Number of input channels.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """
#
#     def __init__(self, dim, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
#                  drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
#         super().__init__()
#         self.dim = dim
#         self.depth = depth
#         self.window_size = window_size
#         self.use_checkpoint = use_checkpoint
#         self.shift_size = window_size // 2
#
#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(
#                 dim=dim,
#                 num_heads=num_heads,
#                 window_size=window_size,
#                 shift_size=0 if (i % 2 == 0) else self.shift_size,
#                 mlp_ratio=mlp_ratio,
#                 qkv_bias=qkv_bias,
#                 drop=drop,
#                 attn_drop=attn_drop,
#                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                 norm_layer=norm_layer)
#             for i in range(depth)])
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#
#     def create_mask(self, x, H, W):
#         # calculate attention mask for SW-MSA
#         # 保证Hp和Wp是window_size的整数倍
#         Hp = int(np.ceil(H / self.window_size)) * self.window_size
#         Wp = int(np.ceil(W / self.window_size)) * self.window_size
#         # 拥有和feature map一样的通道排列顺序，方便后续window_partition
#         img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
#         h_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         w_slices = (slice(0, -self.window_size),
#                     slice(-self.window_size, -self.shift_size),
#                     slice(-self.shift_size, None))
#         cnt = 0
#         for h in h_slices:
#             for w in w_slices:
#                 img_mask[:, h, w, :] = cnt
#                 cnt += 1
#
#         mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
#         mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
#         attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
#         # [nW, Mh*Mw, Mh*Mw]
#         attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         return attn_mask
#
#     def forward(self, x, H, W):
#         attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
#         for blk in self.blocks:
#             blk.H, blk.W = H, W
#             if not torch.jit.is_scripting() and self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x, attn_mask)
#             else:
#                 x = blk(x, attn_mask)
#         if self.downsample is not None:
#             x = self.downsample(x, H, W)
#             H, W = (H + 1) // 2, (W + 1) // 2
#
#         return x, H, W
#
#
# class SwinTransformer(nn.Module):
#     r""" Swin Transformer
#         A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
#           https://arxiv.org/pdf/2103.14030
#
#     Args:
#         patch_size (int | tuple(int)): Patch size. Default: 4
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         embed_dim (int): Patch embedding dimension. Default: 96
#         depths (tuple(int)): Depth of each Swin Transformer layer.
#         num_heads (tuple(int)): Number of attention heads in different layers.
#         window_size (int): Window size. Default: 7
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
#         qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
#         drop_rate (float): Dropout rate. Default: 0
#         attn_drop_rate (float): Attention dropout rate. Default: 0
#         drop_path_rate (float): Stochastic depth rate. Default: 0.1
#         norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
#         patch_norm (bool): If True, add normalization after patch embedding. Default: True
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
#     """
#
#     def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
#                  embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
#                  window_size=7, mlp_ratio=4., qkv_bias=True,
#                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
#                  norm_layer=nn.LayerNorm, patch_norm=True,
#                  use_checkpoint=False, **kwargs):
#         super().__init__()
#
#         self.num_classes = num_classes
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.patch_norm = patch_norm
#         # stage4输出特征矩阵的channels
#         self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
#         self.mlp_ratio = mlp_ratio
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#
#         # build layers
#         self.layers = nn.ModuleList()
#         for i_layer in range(self.num_layers):
#             # 注意这里构建的stage和论文图中有些差异
#             # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
#             layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
#                                 depth=depths[i_layer],
#                                 num_heads=num_heads[i_layer],
#                                 window_size=window_size,
#                                 mlp_ratio=self.mlp_ratio,
#                                 qkv_bias=qkv_bias,
#                                 drop=drop_rate,
#                                 attn_drop=attn_drop_rate,
#                                 drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
#                                 norm_layer=norm_layer,
#                                 downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
#                                 use_checkpoint=use_checkpoint)
#             self.layers.append(layers)
#
#         self.norm = norm_layer(self.num_features)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
#
#         self.apply(self._init_weights)
#
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             nn.init.trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def forward(self, x):
#         # x: [B, L, C]
#         x, H, W = self.patch_embed(x)
#         x = self.pos_drop(x)
#
#         '''
#         for layer in self.layers:
#             x, H, W = layer(x, H, W)
#         '''
#         x,H,W = self.layers[0](x,H,W)
#         f0 = x
#         x, H, W = self.layers[1](x, H, W)
#         f1 = x
#         x, H, W = self.layers[2](x, H, W)
#         f2 = x
#         x, H, W = self.layers[3](x, H, W)
#         f3 = x
#
#         x = self.norm(x)  # [B, L, C]
#         x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
#         avg = torch.flatten(x, 1)
#         out = self.head(avg)
#
#         feats = {}
#         feats["feats"] = [f0, f1, f2, f3]
#         feats["pooled_feat"] = avg
#
#         return out
#
#
# def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
#     # trained ImageNet-1K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=7,
#                             embed_dim=96,
#                             depths=(2, 2, 6, 2),
#                             num_heads=(3, 6, 12, 24),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
#
# def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
#     # trained ImageNet-1K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=7,
#                             embed_dim=96,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(3, 6, 12, 24),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
# @register_model
# def swin_base_patch4_window7_224(num_classes: int = 7,pretrained=False, **kwargs):
#     # trained ImageNet-1K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=7,
#                             embed_dim=128,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(4, 8, 16, 32),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
#
# def swin_base_patch4_window12_384(num_classes: int = 1000, **kwargs):
#     # trained ImageNet-1K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=12,
#                             embed_dim=128,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(4, 8, 16, 32),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
#
# def swin_base_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
#     # trained ImageNet-22K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=7,
#                             embed_dim=128,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(4, 8, 16, 32),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
#
# def swin_base_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
#     # trained ImageNet-22K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=12,
#                             embed_dim=128,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(4, 8, 16, 32),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
#
# def swin_large_patch4_window7_224_in22k(num_classes: int = 21841, **kwargs):
#     # trained ImageNet-22K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=7,
#                             embed_dim=192,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(6, 12, 24, 48),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
#
# def swin_large_patch4_window12_384_in22k(num_classes: int = 21841, **kwargs):
#     # trained ImageNet-22K
#     # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
#     model = SwinTransformer(in_chans=3,
#                             patch_size=4,
#                             window_size=12,
#                             embed_dim=192,
#                             depths=(2, 2, 18, 2),
#                             num_heads=(6, 12, 24, 48),
#                             num_classes=num_classes,
#                             **kwargs)
#     return model
#
# """
# original code from rwightman:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# """
# from functools import partial
# from collections import OrderedDict
# from thop import profile
# import torch
# import torch.nn as nn
#
#
# def drop_path(x, drop_prob: float = 0., training: bool = False):
#     """
#     Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
#     This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
#     the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
#     See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
#     changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
#     'survival rate' as the argument.
#     """
#     if drop_prob == 0. or not training:
#         return x
#     keep_prob = 1 - drop_prob
#     shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
#     random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
#     random_tensor.floor_()  # binarize
#     output = x.div(keep_prob) * random_tensor
#     return output
#
#
# class DropPath(nn.Module):
#     """
#     Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
#     """
#     def __init__(self, drop_prob=None):
#         super(DropPath, self).__init__()
#         self.drop_prob = drop_prob
#
#     def forward(self, x):
#         return drop_path(x, self.drop_prob, self.training)
#
#
# class PatchEmbed(nn.Module):
#     """
#     2D Image to Patch Embedding
#     """
#     def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
#         super().__init__()
#         img_size = (img_size, img_size)
#         patch_size = (patch_size, patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#
#         self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
#         self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
#
#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#
#         # flatten: [B, C, H, W] -> [B, C, HW]
#         # transpose: [B, C, HW] -> [B, HW, C]
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         x = self.norm(x)
#         return x
#
#
# class Attention(nn.Module):
#     def __init__(self,
#                  dim,   # 输入token的dim
#                  num_heads=8,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop_ratio=0.,
#                  proj_drop_ratio=0.):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)
#
#     def forward(self, x):
#         # [batch_size, num_patches + 1, total_embed_dim]
#         B, N, C = x.shape
#
#         # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
#         # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
#         # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#
#         # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
#         # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
#         # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class Mlp(nn.Module):
#     """
#     MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x
#
#
# class Block(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  mlp_ratio=4.,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  drop_ratio=0.,
#                  attn_drop_ratio=0.,
#                  drop_path_ratio=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm):
#         super(Block, self).__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
#
#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x
#
#
# class VisionTransformer(nn.Module):
#     def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
#                  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
#                  qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
#                  attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
#                  act_layer=None):
#         """
#         Args:
#             img_size (int, tuple): input image size
#             patch_size (int, tuple): patch size
#             in_c (int): number of input channels
#             num_classes (int): number of classes for classification head
#             embed_dim (int): embedding dimension
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             distilled (bool): model includes a distillation token and head as in DeiT models
#             drop_ratio (float): dropout rate
#             attn_drop_ratio (float): attention dropout rate
#             drop_path_ratio (float): stochastic depth rate
#             embed_layer (nn.Module): patch embedding layer
#             norm_layer: (nn.Module): normalization layer
#         """
#         super(VisionTransformer, self).__init__()
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.num_tokens = 2 if distilled else 1
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         act_layer = act_layer or nn.GELU
#
#         self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
#         self.pos_drop = nn.Dropout(p=drop_ratio)
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
#         self.blocks = nn.Sequential(*[
#             Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                   drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
#                   norm_layer=norm_layer, act_layer=act_layer)
#             for i in range(depth)
#         ])
#         self.norm = norm_layer(embed_dim)
#
#         # Representation layer
#         if representation_size and not distilled:
#             self.has_logits = True
#             self.num_features = representation_size
#             self.pre_logits = nn.Sequential(OrderedDict([
#                 ("fc", nn.Linear(embed_dim, representation_size)),
#                 ("act", nn.Tanh())
#             ]))
#         else:
#             self.has_logits = False
#             self.pre_logits = nn.Identity()
#
#         # Classifier head(s)
#         self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
#         self.head_dist = None
#         if distilled:
#             self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
#
#         # Weight init
#         nn.init.trunc_normal_(self.pos_embed, std=0.02)
#         if self.dist_token is not None:
#             nn.init.trunc_normal_(self.dist_token, std=0.02)
#
#         nn.init.trunc_normal_(self.cls_token, std=0.02)
#         self.apply(_init_vit_weights)
#
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token'}
#     def forward_features(self, x):
#         # [B, C, H, W] -> [B, num_patches, embed_dim]
#         x = self.patch_embed(x)  # [B, 196, 768]
#         # [1, 1, 768] -> [B, 1, 768]
#         cls_token = self.cls_token.expand(x.shape[0], -1, -1)
#         if self.dist_token is None:
#             x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
#         else:
#             x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
#
#         x = self.pos_drop(x + self.pos_embed)
#         x = self.blocks(x)
#         x = self.norm(x)
#         if self.dist_token is None:
#             return self.pre_logits(x[:, 0])
#         else:
#             return x[:, 0], x[:, 1]
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         if self.head_dist is not None:
#             x, x_dist = self.head(x[0]), self.head_dist(x[1])
#             if self.training and not torch.jit.is_scripting():
#                 # during inference, return the average of both classifier predictions
#                 return x, x_dist
#             else:
#                 return (x + x_dist) / 2
#         else:
#             x = self.head(x)
#         return x
#
#
# def _init_vit_weights(m):
#     """
#     ViT weight initialization
#     :param m: module
#     """
#     if isinstance(m, nn.Linear):
#         nn.init.trunc_normal_(m.weight, std=.01)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode="fan_out")
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)
#     elif isinstance(m, nn.LayerNorm):
#         nn.init.zeros_(m.bias)
#         nn.init.ones_(m.weight)
#
# @register_model
# def vit_base_patch16_224(num_classes: int = 3, pretrained=False, **kwargs):
#     """
#     ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     weights ported from official Google JAX impl:
#     链接: https://pan.baidu.com/s/1zqb08naeP0RPqqfSXfkB2EA  密码: eu9f
#     """
#     model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=768,
#                               depth=12,
#                               num_heads=12,
#                               representation_size=None,
#                               num_classes=num_classes)
#     return model
#
#
# def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
#     """
#     ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     weights ported from official Google JAX impl:
#     https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth
#     """
#     model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=768,
#                               depth=12,
#                               num_heads=12,
#                               representation_size=768 if has_logits else None,
#                               num_classes=num_classes)
#     return model
#
# @register_model
# def vit_base_patch32_224(num_classes: int = 3, pretrained=False, **kwargs):
#     """
#     ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     weights ported from official Google JAX impl:
#     链接: https://pan.baidu.com/s/1hCv0U8pQomwAtHBYc4hmZg  密码: s5hl
#     """
#     model = VisionTransformer(img_size=224,
#                               patch_size=32,
#                               embed_dim=768,
#                               depth=12,
#                               num_heads=12,
#                               representation_size=None,
#                               num_classes=num_classes
#                               )
#     return model
#
#
# def vit_base_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
#     """
#     ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     weights ported from official Google JAX impl:
#     https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth
#     """
#     model = VisionTransformer(img_size=224,
#                               patch_size=32,
#                               embed_dim=768,
#                               depth=12,
#                               num_heads=12,
#                               representation_size=768 if has_logits else None,
#                               num_classes=num_classes)
#     return model
#
#
# def vit_large_patch16_224(num_classes: int = 1000):
#     """
#     ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     weights ported from official Google JAX impl:
#     链接: https://pan.baidu.com/s/1cxBgZJJ6qUWPSBNcE4TdRQ  密码: qqt8
#     """
#     model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=1024,
#                               depth=24,
#                               num_heads=16,
#                               representation_size=None,
#                               num_classes=num_classes)
#     return model
#
#
# def vit_large_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):
#     """
#     ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     weights ported from official Google JAX impl:
#     https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth
#     """
#     model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=1024,
#                               depth=24,
#                               num_heads=16,
#                               representation_size=1024 if has_logits else None,
#                               num_classes=num_classes)
#     return model
#
#
# def vit_large_patch32_224_in21k(num_classes: int = 21843, has_logits: bool = True):
#     """
#     ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     weights ported from official Google JAX impl:
#     https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth
#     """
#     model = VisionTransformer(img_size=224,
#                               patch_size=32,
#                               embed_dim=1024,
#                               depth=24,
#                               num_heads=16,
#                               representation_size=1024 if has_logits else None,
#                               num_classes=num_classes)
#     return model
#
#
# def vit_huge_patch14_224_in21k(num_classes: int = 21843, has_logits: bool = True):
#     """
#     ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
#     NOTE: converted weights not currently available, too large for github release hosting.
#     """
#     model = VisionTransformer(img_size=224,
#                               patch_size=14,
#                               embed_dim=1280,
#                               depth=32,
#                               num_heads=16,
#                               representation_size=1280 if has_logits else None,
#                               num_classes=num_classes)
#     return model

# # Copyright (c) 2015-present, Facebook, Inc.
# # All rights reserved.
# import torch
# import torch.nn as nn
# from functools import partial
#
# from timm.models.vision_transformer import VisionTransformer, _cfg
# from timm.models.registry import register_model
# from timm.models.layers import trunc_normal_
#
#
# __all__ = [
#     'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
#     'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
#     'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
#     'deit_base_distilled_patch16_384',
# ]
#
#
# class DistilledVisionTransformer(VisionTransformer):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
#         num_patches = self.patch_embed.num_patches
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
#         self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
#
#         trunc_normal_(self.dist_token, std=.02)
#         trunc_normal_(self.pos_embed, std=.02)
#         self.head_dist.apply(self._init_weights)
#
#     def forward_features(self, x):
#         # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#         # with slight modifications to add the dist_token
#         B = x.shape[0]
#         x = self.patch_embed(x)
#
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
#         dist_token = self.dist_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, dist_token, x), dim=1)
#
#         x = x + self.pos_embed
#         x = self.pos_drop(x)
#
#         for blk in self.blocks:
#             x = blk(x)
#
#         x = self.norm(x)
#         return x[:, 0], x[:, 1]
#
#     def forward(self, x):
#         x, x_dist = self.forward_features(x)
#         x = self.head(x)
#         x_dist = self.head_dist(x_dist)
#         if self.training:
#             return x, x_dist
#         else:
#             # during inference, return the average of both classifier predictions
#             return (x + x_dist) / 2
#
#
# @register_model
# def deit_tiny_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
#
#
# @register_model
# def deit_small_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
#
#
# @register_model
# def deit_base_patch16_224(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
#
#
# @register_model
# def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
#     model = DistilledVisionTransformer(
#         patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
#
#
# @register_model
# def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
#     model = DistilledVisionTransformer(
#         patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
#
#
# @register_model
# def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
#     model = DistilledVisionTransformer(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
#
#
# @register_model
# def deit_base_patch16_384(pretrained=False, **kwargs):
#     model = VisionTransformer(
#         img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
#
#
# @register_model
# def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
#     model = DistilledVisionTransformer(
#         img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     model.default_cfg = _cfg()
#     if pretrained:
#         checkpoint = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
#             map_location="cpu", check_hash=True
#         )
#         model.load_state_dict(checkpoint["model"])
#     return model
