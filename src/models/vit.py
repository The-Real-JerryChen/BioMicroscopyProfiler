import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import timm
import numpy as np
from typing import Optional, Tuple
from typing import Dict
import math
class Normalizer(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels
        return pixels / 255.0
    
class ViTConfig(PretrainedConfig):
    model_type = "vit"
    def __init__(
        self,
        model_type: str = 'small', 
        in_chans: int = 6,
        img_size: int = 224,
        patch_size: int = 16,
        use_cls_token: bool = True,
        pooling_type: str = 'cls',  
        init_pos_embed_type: str = 'sincos',  
        ge_token: int = 0,
        ge_token_dim: int = 0,  
        drop_path_rate: float = 0.,
        pretrained = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.in_chans = in_chans
        self.img_size = img_size
        self.patch_size = patch_size
        self.use_cls_token = use_cls_token
        self.pooling_type = pooling_type
        self.init_pos_embed_type = init_pos_embed_type
        self.ge_token = ge_token
        self.ge_token_dim = ge_token_dim
        self.drop_path_rate = drop_path_rate
        self.pretrained = pretrained
        if model_type == 'small':
            self.embed_dim = 384
            self.num_heads = 6
            self.depth = 12
            self.mlp_ratio = 4
            self.pretrained_model = 'vit_small_patch16_224_in21k'
        elif model_type == 'base':
            self.embed_dim = 768
            self.num_heads = 12
            self.depth = 12
            self.mlp_ratio = 4
            self.pretrained_model = 'vit_base_patch16_224_in21k'
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if not use_cls_token and pooling_type == 'cls':
            raise ValueError("Cannot use cls token pooling when use_cls_token is False")

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)  
    omega /= (embed_dim // 2)  
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)  # [M, D/2]
    emb_cos = np.cos(out)  # [M, D/2]
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # [M, D]
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True, ge_token=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    num_special_tokens = (1 if cls_token else 0) + ge_token
    if num_special_tokens > 0:
        pos_embed = np.concatenate([np.zeros([num_special_tokens, embed_dim]), pos_embed], axis=0)
    
    return pos_embed

class ViTModel(PreTrainedModel):
    config_class = ViTConfig
    base_model_prefix = "vit"
    
    def __init__(self, config: ViTConfig):
        super().__init__(config)
        self.config = config
        
        self.backbone = timm.create_model(
            config.pretrained_model,
            pretrained=config.pretrained,
            img_size=config.img_size,
            in_chans=config.in_chans,
            patch_size=config.patch_size,
            num_classes=0,
            drop_path_rate=config.drop_path_rate,
            dynamic_img_size=True
        )
        
        self.patch_embed = self.backbone.patch_embed
        
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            nn.init.normal_(self.cls_token, std=1e-6)
        else:
            self.cls_token = None
            
        if config.ge_token > 0:
            self.domain_proj = nn.Linear(config.ge_token_dim, config.embed_dim * config.ge_token)
            nn.init.normal_(self.domain_proj.weight, std=0.02)
            nn.init.zeros_(self.domain_proj.bias)


        num_patches = (config.img_size // config.patch_size) ** 2
        if config.init_pos_embed_type == 'sincos':
            pos_embed = get_2d_sincos_pos_embed(
                config.embed_dim,
                config.img_size // config.patch_size,
                cls_token= config.use_cls_token,
                ge_token = config.ge_token
            )
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
            self.register_buffer('pos_embed', pos_embed)
        else:
            num_special_tokens = (1 if config.use_cls_token else 0) + config.ge_token
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + num_special_tokens, config.embed_dim)
            )
            nn.init.normal_(self.pos_embed, std=0.02)
        
        self.blocks = self.backbone.blocks
        self.norm = self.backbone.norm
        
        self.input_norm = torch.nn.Sequential(
            Normalizer(),
            nn.InstanceNorm2d(
                num_features=self.config.in_chans,  
                affine=False, 
                track_running_stats=False
            )
        )
        if not config.pretrained:
            self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
            
    def load_pretrained_weights(self):
        pretrained_dict = self.backbone.state_dict()
        model_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if k in model_dict and 'patch_embed.proj' not in k:
                if v.shape == model_dict[k].shape:
                    model_dict[k] = v
        
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded pretrained weights from {self.config.pretrained_model}")
        
    def interpolate_pos_encoding(self, x, H, W):
        B, N, D = x.shape
        num_special = 1 if self.config.use_cls_token else 0 
        num_special += self.config.ge_token
        special_pos = self.pos_embed[:, :num_special]          # [1, num_special, D]
        patch_pos   = self.pos_embed[:, num_special:]          # [1, num_patches, D]

        orig_size = int(math.sqrt(patch_pos.shape[1]))
        new_h = H // self.config.patch_size
        new_w = W // self.config.patch_size

        patch_pos = patch_pos.reshape(1, orig_size, orig_size, D).permute(0, 3, 1, 2)

        patch_pos = F.interpolate(patch_pos, size=(new_h, new_w), mode='bicubic', align_corners=False)

        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_h * new_w, D)

        return torch.cat((special_pos, patch_pos), dim=1)  # [1, num_special+new_h*new_w, D]

        
    def forward(
        self,
        pixel_values: torch.Tensor,
        gene_features: torch.Tensor = None,  
        return_all_tokens: bool = False,  
    ):
        x = self.input_norm(pixel_values)
        x = self.patch_embed(x)
        B, H, W, D = x.shape
        x = x.contiguous().view(B, H*W, D)
        
        if self.config.use_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            if self.config.ge_token:
                if gene_features is None:
                    raise ValueError("When ge_token is True, gene_features must be provided.")
                ge_tokens = self.domain_proj(gene_features)  #[B,D]
                ge_tokens = ge_tokens.reshape(x.shape[0], self.config.ge_token, self.config.embed_dim)  # [B, ge_token, D]

                x = torch.cat([cls_token, ge_tokens, x], dim=1)  # [B, 1+1+N, D]
            else:
                x = torch.cat([cls_token, x], dim=1)
        
        if pixel_values.shape[2] != self.config.img_size:
            # print(f"pixel_values.shape: {pixel_values.shape}")
            H_image, W_image = pixel_values.shape[2:]
            pos_embed = self.interpolate_pos_encoding(x, H_image, W_image)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        if not return_all_tokens: 
            # Pooling
            if self.config.pooling_type == 'cls' and self.config.use_cls_token:
                x = x[:, 0]

            else:
                start_idx = (1 if self.config.use_cls_token else 0)
                if self.config.ge_token:
                    start_idx += self.config.ge_token
                x = x[:, start_idx:].mean(dim=1)
        
        return x
    

        
    def forward_with_mask(self, pixel_values, gene_features=None, mask_ratio=0.75):
        x = self.patch_embed(pixel_values)
        B, H, W, D = x.shape
        x = x.contiguous().view(B, H*W, D)
        B, N, D = x.shape
        # tokens = []
        if self.config.use_cls_token:
            cls_token = self.cls_token.expand(B, -1, -1)
            if self.config.ge_token > 0:
                assert gene_features is not None
                ge = self.domain_proj(gene_features)
                ge_tokens = ge.reshape(B, self.config.ge_token, self.config.embed_dim)
                
                x = torch.cat([cls_token, ge_tokens, x], dim=1)  # [B, S+N, D]
            else:
                x = torch.cat([cls_token, x], dim=1)  # [B, S+N, D]
        else:
            x = torch.cat([x], dim=1)  # [B, N, D]
        x = x + self.pos_embed
        S = (1 if self.config.use_cls_token else 0) + self.config.ge_token
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # gather kept
        kept = torch.gather(
            x[:, S:],
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        x_masked = torch.cat([cls_token, ge_tokens, kept], dim=1) if self.config.ge_token > 0 else torch.cat([cls_token, kept], dim=1)
        # build mask
        mask = torch.ones(B, S + N, device=x.device)
        mask[:, :S] = 0
        flat = torch.ones(B, N, device=x.device)
        flat.scatter_(1, ids_keep, 0)
        mask[:, S:] = flat
        for blk in self.blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm(x_masked)
        return x_masked, mask, ids_restore