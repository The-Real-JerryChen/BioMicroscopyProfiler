import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from typing import Optional, Dict, Tuple

class Normalizer(torch.nn.Module):
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels
        return pixels / 255.0
    
class Block(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x
    
class MAEDecoder(nn.Module):
    def __init__(self, config, decoder_dim=256, depth=8, num_heads=16):
        super().__init__()
        self.S = (1 if config.use_cls_token else 0) + config.ge_token
        self.P = (config.img_size // config.patch_size) ** 2
        self.patch_size = config.patch_size
        self.in_chans   = config.in_chans

        self.decoder_embed = nn.Linear(config.embed_dim, decoder_dim, bias=True)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed     = nn.Parameter(torch.zeros(1, self.S + self.P, decoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([Block(decoder_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, self.patch_size**2 * self.in_chans)

    def forward(self, x, ids_restore):
        """
        x:           [B, S + keep, encoder_dim]
        ids_restore: [B, P]
        """
        B, L_vis, _ = x.shape
        x = self.decoder_embed(x)
        D_dec = x.size(-1)

        x_special = x[:, :self.S, :]
        x_patches = x[:, self.S:, :]
        keep = x_patches.size(1)

        mask_tokens = self.mask_token.expand(B, self.P - keep, D_dec)
        x_cat = torch.cat([x_patches, mask_tokens], dim=1)

        idx = ids_restore.unsqueeze(-1).expand(-1, -1, D_dec)
        x_patches_full = torch.gather(x_cat, dim=1, index=idx)

        x_full = torch.cat([x_special, x_patches_full], dim=1)

        x_full = x_full + self.pos_embed
        for blk in self.blocks:
            x_full = blk(x_full)
        x_full = self.norm(x_full)

        x_p = self.decoder_pred(x_full[:, self.S:])

        p = self.patch_size
        grid = int(self.P ** 0.5)
        c = self.in_chans
        x_p = x_p.view(B, self.P, p, p, c)
        x_p = x_p.permute(0,4,1,2,3).reshape(B, c, grid*p, grid*p)
        return x_p


class ViTMAE(nn.Module):
    def __init__(self, encoder, mask_ratio=0.75, norm_pix_loss=True, supervised_loss_fn=None, supervised_weight=0.0, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.decoder = MAEDecoder(encoder.config)
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.supervised_loss_fn = supervised_loss_fn
        self.supervised_weight = supervised_weight
        self.input_norm = torch.nn.Sequential(
            Normalizer(),
            nn.InstanceNorm2d(
                num_features=6,  
                affine=False, 
                track_running_stats=False
            )
        )
    def patchify(self, imgs):
        p = self.encoder.config.patch_size
        B, C, H, W = imgs.shape
        h, w = H//p, W//p
        x = imgs.reshape(B, C, h, p, w, p).permute(0,2,4,1,3,5)
        return x.reshape(B, h*w, C*p*p)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        images = batch["imgs"]
        images = self.input_norm(images)
        gene_features = batch.get("gene_features", None)
        batch_matrix = batch.get("batch_matrix", None)
        kwargs = {"gene_features": gene_features, "mask_ratio": self.mask_ratio}
        
        latent, mask, ids = self.encoder.forward_with_mask(images, **kwargs)
        S = (1 if self.encoder.config.use_cls_token else 0) + self.encoder.config.ge_token
        global_repr = latent[:, S:].mean(dim=1)  
        
        pred = self.decoder(latent, ids)
        if self.norm_pix_loss:
            target = self.patchify(images)
            mean = target.mean(-1, keepdim=True)
            std = target.std(-1, keepdim=True)
            target = (target - mean) / (std + 1e-6)
            
            pred_p = self.patchify(pred)
            loss = (pred_p - target).pow(2).mean(-1)
            S = (1 if self.encoder.config.use_cls_token else 0) + self.encoder.config.ge_token
            mask_p = mask[:, S:]
            loss = (loss * mask_p).sum() / mask_p.sum()
        else:
            loss = F.mse_loss(pred, images)
            
        losses_dict = {"mae_loss": loss}
        total_loss = loss
        
        if self.supervised_weight >0 and batch_matrix is not None:

            sup_loss = self.supervised_loss_fn(global_repr, batch_matrix)
            total_loss = total_loss + self.supervised_weight * sup_loss
            losses_dict["supervised_loss"] = sup_loss
            
        losses_dict["total_loss"] = total_loss
        
        return {'loss': total_loss, 'reconstruction': pred, 'mask': mask, **losses_dict}