import os
import warnings
from functools import partial

import huggingface_hub as hf
import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import Swin_V2_B_Weights, swin_v2_b

# mamba-ssm
# causal-conv1d
# rope

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="",
        category=FutureWarning
    )
    from model.mamba import (VisionMamba,
                             vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2)


class DenseNet121_v0(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, n_class):
        super(DenseNet121_v0, self).__init__()
        self.densenet121 = torchvision.models.densenet121()
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, n_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class dense121_mcs(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """

    def __init__(self, n_class):
        super(dense121_mcs, self).__init__()

        self.densenet121 = torchvision.models.densenet121()
        num_ftrs = self.densenet121.classifier.in_features

        A_model = DenseNet121_v0(n_class=n_class)
        self.featureA = A_model
        self.classA = A_model.densenet121.features

        B_model = DenseNet121_v0(n_class=n_class)
        self.featureB = B_model
        self.classB = B_model.densenet121.features

        C_model = DenseNet121_v0(n_class=n_class)
        self.featureC = C_model
        self.classC = C_model.densenet121.features

        self.combine1 = nn.Sequential(
            nn.Linear(n_class * 4, n_class),
            nn.Sigmoid()
        )

        self.combine2 = nn.Sequential(
            nn.Linear(num_ftrs * 3, n_class),
            nn.Sigmoid()
        )

    def forward(self, x, y, z):
        x1 = self.featureA(x)
        y1 = self.featureB(y)
        z1 = self.featureC(z)
        x2 = self.classA(x)
        x2 = F.relu(x2, inplace=True)
        x2 = F.adaptive_avg_pool2d(x2, (1, 1)).view(x2.size(0), -1)
        y2 = self.classB(y)
        y2 = F.relu(y2, inplace=True)
        y2 = F.adaptive_avg_pool2d(y2, (1, 1)).view(y2.size(0), -1)
        z2 = self.classC(z)
        z2 = F.relu(z2, inplace=True)
        z2 = F.adaptive_avg_pool2d(z2, (1, 1)).view(z2.size(0), -1)

        combine = torch.cat((x2.view(x2.size(0), -1),
                             y2.view(y2.size(0), -1),
                             z2.view(z2.size(0), -1)), 1)
        combine = self.combine2(combine)

        combine3 = torch.cat((x1.view(x1.size(0), -1),
                              y1.view(y1.size(0), -1),
                              z1.view(z1.size(0), -1),
                              combine.view(combine.size(0), -1)), 1)

        combine3 = self.combine1(combine3)

        return x1, y1, z1, combine, combine3


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def get_EyeQ():
    eyeq = dense121_mcs(3)
    # https://github.com/HzFu/EyeQ
    # https://1drv.ms/u/s!ArBRrL8ao6jznU6RCbo60oStjPWZ?e=qQmzST
    eyeq_pretrain = torch.load(
        os.path.expanduser("/home/dell/lyh/pretrained/fundus/DenseNet121_v3_v1.tar"),
        map_location=torch.device("cpu")
    )
    eyeq.load_state_dict(eyeq_pretrain["state_dict"])
    del eyeq_pretrain
    return eyeq


def interpolate_pos_embed(model, checkpoint_model):
    """
    Interpolate position embeddings for high-resolution.
    References:
    DeiT: https://github.com/facebookresearch/deit
    """
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def get_RETFound():
    class VisionTransformerNoHead(VisionTransformer):
        def __init__(self, global_pool=False, **kwargs):
            super().__init__(global_pool, **kwargs)

        def forward_head(
            self, x: torch.Tensor, pre_logits: bool = False
        ) -> torch.Tensor:
            x = x.view(1, -1)
            return x

    def vit_large_patch16_no_head(**kwargs):
        model = VisionTransformerNoHead(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **kwargs,
        )
        return model

    model = vit_large_patch16_no_head(drop_path_rate=0.2, global_pool=True)
    path = hf.try_to_load_from_cache("YukunZhou/RETFound_mae_natureCFP", "RETFound_mae_natureCFP.pth")
    # path = f"/home/app/data/retipioneeralgorithm/model/RETFound_mae_natureCFP.pth"
    checkpoint = hf.load_state_dict_from_file(path, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)

    assert set(msg.missing_keys) == {
        "head.weight",
        "head.bias",
        "fc_norm.weight",
        "fc_norm.bias",
    }
    del checkpoint_model
    return model


def get_VimS():
    class VimFeat(nn.Module):
        def __init__(self, vim: VisionMamba):
            super().__init__()
            self.vim = vim

        def forward(self, x):
            return self.vim(x, return_features=True)

    model = vim_small_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2()
    sd = torch.load("/home/dell/lyh/pretrained/vim_s_midclstok_ft_81p6acc.pth", weights_only=False)
    model.load_state_dict(sd["model"])
    return VimFeat(model)


def get_SwinB():
    model = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
    model.head = nn.Identity()
    return model
