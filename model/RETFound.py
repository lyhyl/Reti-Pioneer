from functools import partial

import torch
import torch.nn as nn

from RETFound_MAE.models_vit import VisionTransformer
from RETFound_MAE.util.pos_embed import interpolate_pos_embed


def get_RETFound_model():
    class VisionTransformerNoHead(VisionTransformer):
        def __init__(self, global_pool=False, **kwargs):
            super().__init__(global_pool, **kwargs)

        def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
            return x

    def vit_large_patch16_no_head(**kwargs):
        model = VisionTransformerNoHead(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        return model
    
    model = vit_large_patch16_no_head(
        drop_path_rate=0.2,
        global_pool=True
    )
    # load RETFound weights
    checkpoint = torch.load('ckpt/RETFound_cfp_weights.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    del checkpoint_model
    return model
