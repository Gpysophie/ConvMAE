# Copyright (c) 2022 Alpha-VL
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import pdb
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from vision_transformer import HybridEmbed, PatchEmbed2d, CBlock2d, Block, Mlp


class FiLM(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.lin_gamma = nn.Linear(n_in, n_out)
        self.lin_beta = nn.Linear(n_in, n_out)

    def forward(self, c, x):
        gamma = self.lin_gamma(c)
        beta = self.lin_beta(c)
        return gamma * x + beta


class ConvViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=False, head_type='MLP',
                 meta_film_flag=False):
        super().__init__()
        self.patch_size = patch_size
        self.global_pool = global_pool
        self.head_type = head_type
        self.meta_film_flag = meta_film_flag

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed1 = PatchEmbed2d(
            img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed2d(
            img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed2d(
            img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])
        num_patches = self.patch_embed3.num_patches
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim[2]))
        self.pos_drop = nn.Dropout(p=drop_rate)

        meta_in_ch = 5
        self.cls_token = nn.Sequential(nn.Linear(meta_in_ch, 64), nn.GELU(), nn.Linear(64, embed_dim[2]))
        self.meta_film_token = nn.Sequential(nn.Linear(meta_in_ch, 64), nn.GELU(), nn.Linear(64, embed_dim[2]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        self.blocks1 = nn.ModuleList([
            CBlock2d(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock2d(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2], num_heads=num_heads, mlp_ratio=mlp_ratio[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth[0] + depth[1] + i], norm_layer=norm_layer)
            for i in range(depth[2])])


        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], (self.patch_size[1] * self.patch_size[2], 1), stride=(self.patch_size[1] * self.patch_size[2], 1))
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], (self.patch_size[2], 1), stride=(self.patch_size[2], 1))

        # self.all_stage_decode = nn.Sequential(
        #     nn.Linear(embed_dim[-1], 384),
        #     nn.GELU(),
        #     nn.Linear(384, embed_dim[-1])
        # )

        self.norm = norm_layer(embed_dim[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim[-1])

            del self.norm  # remove the original norm

        self.film_meta = FiLM(embed_dim[2], embed_dim[2])

        # Classifier head
        hidden_mlp = 768
        # self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()
        if head_type == 'Linear':
            # self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head1 = nn.Linear(embed_dim[-1], 1) if num_classes > 0 else nn.Identity()
            self.head2 = nn.Linear(embed_dim[-1], 1) if num_classes > 0 else nn.Identity()
        elif head_type == 'MLP':
            drop = 0.1
            act_layer = nn.GELU
            self.head1 = nn.Sequential(
                Mlp(in_features=embed_dim[-1], hidden_features=hidden_mlp, out_features=int(hidden_mlp / 2),
                    act_layer=act_layer, drop=drop),
                Mlp(in_features=int(hidden_mlp / 2), hidden_features=int(hidden_mlp / 4), out_features=1,
                    act_layer=act_layer, drop=0)
            )
            self.head2 = nn.Sequential(
                Mlp(in_features=embed_dim[-1], hidden_features=hidden_mlp, out_features=int(hidden_mlp / 2),
                    act_layer=act_layer, drop=drop),
                Mlp(in_features=int(hidden_mlp / 2), hidden_features=int(hidden_mlp / 4), out_features=1,
                    act_layer=act_layer, drop=0)
            )

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, meta):
        B = x.shape[0]
        x = x.unsqueeze(-1)
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        B_tmp, N_tmp, C_tmp = x.shape
        x = self.patch_embed4(x.reshape(B_tmp * N_tmp, C_tmp)).reshape(B_tmp, N_tmp, C_tmp)
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token(meta[:, :]).unsqueeze(1)
        cls_tokens = cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks3:
            x = blk(x)

        x = x[:, 1:, :] + stage1_embed + stage2_embed
        # x = self.all_stage_decode(x)

        if self.global_pool:
            x = x[:, :, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
        return outcome

    def forward(self, x, meta):
        meta = torch.concat((meta[:, :4], meta[:, 5, None]), dim=1)
        x = self.forward_features(x, meta)

        if self.meta_film_flag:
            x_meta = self.meta_film_token(meta)
            x = self.film_meta(x_meta, x)

        x1 = self.head1(x)
        x2 = self.head2(x.detach())
        x = torch.cat((x1, x2), dim=1)
        return x


def convvit_tiny_patch16(**kwargs):
    model = ConvViT(
        embed_dim=[64, 128, 256], depth=[1, 1, 7], num_heads=8, mlp_ratio=[2, 2, 2], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def convvit_small_patch16(**kwargs):
    model = ConvViT(
        embed_dim=[64, 128, 256], depth=[2, 2, 8], num_heads=8, mlp_ratio=[2, 2, 2], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def convvit_base_patch16(**kwargs):
    model = ConvViT(
        embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def convvit_large_patch16(**kwargs):
    model = ConvViT(
        embed_dim=[384, 768, 1024], depth=[4, 4, 12], num_heads=16, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



