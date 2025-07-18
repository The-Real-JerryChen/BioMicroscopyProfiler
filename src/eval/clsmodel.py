import os
import sys
import torch
import torch.nn as nn
from typing import Dict


class CLSModel(nn.Module):
    """
    Supervised Learning Model
    """
    def __init__(
        self, 
        encoder: nn.Module,
        embed_dim: int,
        num_classes: int = 1138,
        supervised_loss_fn = None,
        supervised_weight: float = 0.0,
        **kwargs,
    ):
        """
        """
        super().__init__()
        self.encoder = encoder
        
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.supervised_loss_fn = supervised_loss_fn
        self.supervised_weight = supervised_weight

        
        
        self.classification_loss = nn.CrossEntropyLoss()
    
    
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        imgs = batch["pixels"]
        labels = batch["label"]
        gene_features = batch.get("gene_features", None)
        batch_matrix = batch.get("batch_matrix", None)
        sirna_labels = batch.get("sirna", None)
        
        kwargs = {"gene_features": gene_features} if gene_features is not None else {}
        
        features = self.encoder(imgs, **kwargs)
        
        logits = self.classifier(features)
        
        cls_loss = self.classification_loss(logits, labels)
        
        losses_dict = {"classification_loss": cls_loss}
        total_loss = cls_loss
        
        if self.supervised_weight >0 and batch_matrix is not None:

            sup_loss = self.supervised_loss_fn(features, batch_matrix, sirna_labels)
            total_loss = total_loss + self.supervised_weight * sup_loss
            losses_dict["graph_loss"] = sup_loss
        
        
        losses_dict["total_loss"] = total_loss
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        labels = labels.cpu() if isinstance(labels, torch.Tensor) else labels

        return {"loss": total_loss, "logits": logits, "acc": acc, "labels": labels, **losses_dict}


def load_pretrained_weights(model, dir, ckpt_path):
    ckpt = torch.load(os.path.join(dir, ckpt_path), map_location='cpu')['state_dict']
    new_sd = {}
    model_dict = model.state_dict()
    
    for key, val in ckpt.items():
        if key.startswith('encoder.'):
            new_key = key[len('encoder.'):]
            
            if 'patch_embed.proj.weight' in new_key:
                # From 6 channels to 5 channels: drop channel 4
                # val shape: [384, 6, 16, 16]
                # need to be: [384, 5, 16, 16]
                if val.shape[1] == 6 and model_dict[new_key].shape[1] == 5:
                    # select channels 0,1,2,3,5 (skip index 4)
                    val = val[:, [0,1,2,3,5], :, :]
                    print(f"Adjusted {new_key} from 6 channels to 5 channels (dropped channel 4)")
        
            if new_key in model_dict:
                if val.shape == model_dict[new_key].shape:
                    new_sd[new_key] = val
                else:
                    print(f"Skipping {new_key}: shape mismatch - checkpoint: {val.shape}, model: {model_dict[new_key].shape}")
            else:
                print(f"Skipping {new_key}: not found in current model")
    missing_keys, unexpected_keys = model.load_state_dict(new_sd, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
        
    print(f"Loaded pretrained weights from {ckpt_path}")
    return model