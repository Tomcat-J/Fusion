# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from functools import partial
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import PatchEmbed
# from mae_lite.models.models_vit import Block
from projects.eval_tools.models_vit_rpe import Block, RelativePositionBias
from projects.mae_lite.util.pos_embed import get_2d_sincos_pos_embed
from timm.models import register_model


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        drop_path_rate=0.,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True,
        use_abs_pos_emb=True,
        qkv_bias=False,
        qk_scale=None,
        init_values=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        num_classes = 3,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=12,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_pred_dim=None,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        out_indices=[3, 5, 7, 11],
        use_mean_pooling=True,
        norm_pix_loss=False,
        init_scale=0.001
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.out_indices = out_indices
        self.use_abs_pos_emb = use_abs_pos_emb
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        # if use_abs_pos_emb:
        #     self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        # else:
        #     self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.grid_size, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        edpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=edpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.grid_size if use_rel_pos_bias else None,index = i)
            if i not in self.out_indices else
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=edpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.grid_size if use_rel_pos_bias else None,index = i,lwindow_size=5)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding
        ddpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            [
            Block(
                dim=decoder_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=ddpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=None)
                for i in range(decoder_depth)
            ]
        )
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_norm = norm_layer(decoder_embed_dim)
        decoder_pred_dim = patch_size ** 2 * in_chans if decoder_pred_dim is None else decoder_pred_dim
        self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_pred_dim, bias=True)  # encoder to decoder
        # --------------------------------------------------------------------------
        self.enorm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # Add an additional fully connected layer
        self.additional_fc = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1280), nn.GELU())

        self.head = nn.Linear(1280, num_classes) if num_classes > 0 else nn.Identity()
        if self.pos_embed is not None:
            self._trunc_normal_(self.pos_embed, std=.02)
        self._trunc_normal_(self.cls_token, std=.02)
        if num_classes > 0:
            self._trunc_normal_(self.head.weight, std=.02)


        if num_classes > 0:
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)
        self.out_indices = out_indices
        self.norm_pix_loss = norm_pix_loss
        self.out_indices = out_indices


        self.initialize_weights()

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            if layer_id in self.out_indices:
                rescale(layer.attn.local_attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)
                rescale(layer.attn.proj.weight.data, layer_id + 1)
            else:
                rescale(layer.attn.proj.weight.data, layer_id + 1)
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self._trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            self._trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5), cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)
        self.fix_init_weight()

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         # we use xavier_uniform following official JAX ViT:
    #         torch.nn.init.xavier_uniform_(m.weight)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, ids_shuffle=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        if ids_shuffle is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_shuffle

    def forward_encoder(self, x, mask_ratio, ids_shuffle=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_shuffle = self.random_masking(x, mask_ratio, ids_shuffle)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x,_,_ = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, ids_shuffle

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x,_,_ = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_features(self, x, with_hidden=False, with_attn=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        hiddens = []
        attns = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x, hidden, attn = blk(x, rel_pos_bias=rel_pos_bias, with_hidden=with_hidden, with_attn=with_attn)
            hiddens.append(hidden)
            attns.append(attn)

        x = self.enorm(x)
        if self.fc_norm is not None:
            hiddens.append(self.fc_norm(x) if with_hidden else None)
            t = x[:, 1:, :]
            outcome = self.fc_norm(t.mean(1))
            for i, _ in enumerate(self.blocks):
                if i in self.out_indices:
                    if i==11:
                        outcome = self.additional_fc(outcome)
        else:
            hiddens.append(x if with_hidden else None)
            outcome = x[:, 0]
        attns = None if all([attn is None for attn in attns]) else attns
        hiddens = None if all([hidden is None for hidden in hiddens]) else hiddens
        return outcome, hiddens, attns

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, ids_shuffle=None):
        if self.training:
            latent, mask, ids_restore, ids_shuffle = self.forward_encoder(imgs, mask_ratio, ids_shuffle)
            pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask)
            return loss, pred, mask, ids_shuffle
        else:
            x, _, _ = self.forward_features(imgs)
            x = self.head(x)
            return x



@register_model
def mae_vit_tiny_rpe_patch16(pretrained=False, global_pool=True, **kwargs):
    # the number of heads is changed to 12 from 3, which is different to the original arch.
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=12,
        decoder_embed_dim=96,
        decoder_depth=1,
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=global_pool,
        init_values=0.1,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_abs_pos_emb=False,
        **kwargs
    )
    return model


@register_model
def mae_vit_small_rpe_patch16(pretrained=False, global_pool=True, **kwargs):
    # the number of heads is changed to 12 from 6, which is different to the original arch.
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        decoder_embed_dim=192,
        decoder_depth=1,
        decoder_num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=global_pool,
        init_values=0.1,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_abs_pos_emb=False,
        **kwargs
    )
    return model


@register_model
def mae_vit_base_rpe_patch16(pretrained=False, global_pool=True, **kwargs):
    # for fine-tuning
    # defaults = {
    #     "use_abs_pos_emb": False,
    #     "use_rel_pos_bias": True,
    #     "use_shared_rel_pos_bias": False,
    # }
    # defaults.update(kwargs)
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_mean_pooling=global_pool,
        init_values=0.1,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_abs_pos_emb=False,
        **kwargs
    )
    return model


@register_model
def mae_vit_large_rpe_patch16(pretrained=False, global_pool=True, **kwargs):
    # for fine-tuning
    defaults = {
        "use_abs_pos_emb": False,
        "use_rel_pos_bias": True,
        "use_shared_rel_pos_bias": False,
    }
    defaults.update(kwargs)
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True,
        use_abs_pos_emb=True,
        **defaults
    )
    return model


@register_model
def mae_vit_huge_rpe_patch14(pretrained=False, global_pool=True, **kwargs):
    # for fine-tuning
    defaults = {
        "use_abs_pos_emb": False,
        "use_rel_pos_bias": True,
        "use_shared_rel_pos_bias": False,
    }
    defaults.update(kwargs)
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.1,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=True,
        use_abs_pos_emb=True,
        **defaults
    )
    return model
