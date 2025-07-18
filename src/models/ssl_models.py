import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Dict
from .mae import ViTMAE
import math
import torch.distributed


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 norm_layer: bool = True, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        if norm_layer:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if norm_layer:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class BYOLModel(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module,
        embed_dim: int,
        projector_dim: int = 256, 
        projector_hidden_dim: int = 2048,
        predictor_hidden_dim: int = 2048,
        target_momentum: float = 0.996,
        supervised_loss_fn = None,
        supervised_weight: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        
        self.online_projector = MLP(
            input_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_dim
        )
        
        self.online_predictor = MLP(
            input_dim=projector_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projector_dim
        )
        
        self.target_encoder = deepcopy(encoder)
        self.target_projector = deepcopy(self.online_projector)
        self._stop_gradient(self.target_encoder)
        self._stop_gradient(self.target_projector)
        
        
        self.supervised_loss_fn = supervised_loss_fn
        self.supervised_weight = supervised_weight
        
        self.base_momentum = target_momentum
        total_epochs = kwargs.pop('total_epochs')
        self.total_epochs = total_epochs
        self._update_momentum(0)
    
    def _stop_gradient(self, network: nn.Module):
        for param in network.parameters():
            param.requires_grad = False
    
    
    def _update_momentum(self, current_epoch):
        t = current_epoch
        T = self.total_epochs
        m0 = self.base_momentum
        self.current_momentum = 1 - (1 - m0) * ((math.cos(math.pi * t / T) + 1) / 2)   
    
    def _update_target_network(self):

        for online_param, target_param in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data = (
                self.current_momentum * target_param.data 
                + (1 - self.current_momentum) * online_param.data
            )
        
        for online_param, target_param in zip(
            self.online_projector.parameters(), self.target_projector.parameters()
        ):
            target_param.data = (
                self.current_momentum * target_param.data 
                + (1 - self.current_momentum) * online_param.data
            )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        view1 = batch["view1"]
        view2 = batch["view2"]
        gene_features = batch.get("gene_features", None)
        batch_matrix = batch.get("batch_matrix", None)
        
        kwargs = {"gene_features": gene_features} if gene_features is not None else {}
        embed1 = self.encoder(view1, **kwargs)
        proj1 = self.online_projector(embed1)
        pred1 = self.online_predictor(proj1)
        
        with torch.no_grad():
            embed2 = self.target_encoder(view2, **kwargs)
            proj2 = self.target_projector(embed2)
        
        embed2_online = self.encoder(view2, **kwargs)
        proj2_online = self.online_projector(embed2_online)
        pred2 = self.online_predictor(proj2_online)
        
        with torch.no_grad():
            embed1_target = self.target_encoder(view1, **kwargs)
            proj1_target = self.target_projector(embed1_target)
        
        byol_loss1 = self._byol_loss(pred1, proj2)
        byol_loss2 = self._byol_loss(pred2, proj1_target)
        byol_loss = (byol_loss1 + byol_loss2) / 2
        
        loss = byol_loss
        losses_dict = {"byol_loss": byol_loss}
        
        if self.supervised_weight >0  and batch_matrix is not None:
            sup_loss = self.supervised_loss_fn(embed1, batch_matrix)
            loss = loss + self.supervised_weight * sup_loss
            losses_dict["supervised_loss"] = sup_loss
            losses_dict["total_loss"] = loss
        
        
        return {"loss": loss, **losses_dict}
    
    def _byol_loss(self, online_pred: torch.Tensor, target_proj: torch.Tensor) -> torch.Tensor:

        online_pred = F.normalize(online_pred, dim=-1, p=2)
        target_proj = F.normalize(target_proj, dim=-1, p=2)
        
        return 2 - 2 * (online_pred * target_proj).sum(dim=-1).mean()
    
    def get_embedding(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder(x, **kwargs)


class SimCLRModel(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module,
        embed_dim: int,
        projector_dim: int = 128, 
        projector_hidden_dim: int = 2048,
        temperature: float = 0.1,
        supervised_loss_fn = None,
        supervised_weight: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = MLP(
            input_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_dim
        )
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.supervised_loss_fn = supervised_loss_fn
        self.supervised_weight = supervised_weight
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        view1 = batch["view1"]
        view2 = batch["view2"]
        gene_features = batch.get("gene_features", None)
        batch_matrix = batch.get("batch_matrix", None)
        
        kwargs = {"gene_features": gene_features} if gene_features is not None else {}
        embed1 = self.encoder(view1, **kwargs)
        embed2 = self.encoder(view2, **kwargs)
        
        z1 = self.projector(embed1)
        z2 = self.projector(embed2)
        
        simclr_loss = self._simclr_loss(z1, z2, self.temperature.item())
        
        loss = simclr_loss
        losses_dict = {"simclr_loss": simclr_loss}
        
        if self.supervised_weight >0  and batch_matrix is not None:
            embeddings = (embed1+embed2)/2

            sup_loss = self.supervised_loss_fn(embeddings, batch_matrix)
            loss = loss + self.supervised_weight * sup_loss
            losses_dict["supervised_loss"] = sup_loss
            losses_dict["total_loss"] = loss
        
        return {"loss": loss, **losses_dict}
    
    def _simclr_loss(self, z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        batch_size = z_i.size(0)
        
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        representations = torch.cat([z_i, z_j], dim=0)
        
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device))
        
        for i in range(batch_size):
            mask[i, i + batch_size] = 0
            mask[i + batch_size, i] = 0
        
        negatives = similarity_matrix[mask].view(batch_size * 2, -1)
        
        logits = torch.cat([positives.unsqueeze(1), negatives], dim=1) / temperature
        
        labels = torch.zeros(batch_size * 2, dtype=torch.long, device=z_i.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def get_embedding(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder(x, **kwargs)


class WSLModel(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module,
        embed_dim: int,
        num_classes: int = 1138,
        classifier_hidden_dim: int = 1024,
        dropout: float = 0.1,
        supervised_loss_fn = None,
        supervised_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.encoder = encoder
        

        self.classifier = MLP(
            input_dim=embed_dim,
            hidden_dim=classifier_hidden_dim,
            output_dim=num_classes,
            dropout=dropout
        )
        
        self.supervised_loss_fn = supervised_loss_fn
        self.supervised_weight = supervised_weight

        
        self.apply(self._init_weights)
        
        self.classification_loss = nn.CrossEntropyLoss()
        
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        imgs = batch["imgs"]
        labels = batch["labels"]
        gene_features = batch.get("gene_features", None)
        batch_matrix = batch.get("batch_matrix", None)
        
        kwargs = {"gene_features": gene_features} if gene_features is not None else {}
        
        features = self.encoder(imgs, **kwargs)
        
        logits = self.classifier(features)
        
        cls_loss = self.classification_loss(logits, labels)
        
        losses_dict = {"wsl_loss": cls_loss}
        total_loss = cls_loss
        
        if self.supervised_weight >0 and batch_matrix is not None:

            sup_loss = self.supervised_loss_fn(features, batch_matrix)
            total_loss = total_loss + self.supervised_weight * sup_loss
            losses_dict["supervised_loss"] = sup_loss
        
        
        losses_dict["total_loss"] = total_loss
        
        return {"loss": total_loss, "logits": logits, **losses_dict}


class MoCoV3Model(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module,
        embed_dim: int,
        projector_dim: int = 256,
        projector_hidden_dim: int = 2048,
        predictor_hidden_dim: int = 2048,
        base_momentum: float = 0.99,
        momentum_cosine: bool = True,
        supervised_loss_fn = None,
        supervised_weight: float = 0.0,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.encoder = encoder
        
        self.projector_q = MLP(
            input_dim=embed_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_dim
        )
        
        self.predictor_q = MLP(
            input_dim=projector_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projector_dim
        )
        self.temperature  = temperature
        
        self.encoder_k = deepcopy(encoder)
        self.projector_k = deepcopy(self.projector_q)
        
        self._stop_gradient(self.encoder_k)
        self._stop_gradient(self.projector_k)
        
        self.supervised_loss_fn = supervised_loss_fn
        self.supervised_weight = supervised_weight
        
        self.base_momentum = base_momentum
        self.momentum_cosine = momentum_cosine
        self.total_epochs = kwargs.pop('total_epochs', 300)
        print(f"total_epochs: {self.total_epochs} for MoCo v3")
        self._update_momentum(0)
    
    def _stop_gradient(self, network: nn.Module):
        for param in network.parameters():
            param.requires_grad = False
    
    def _update_momentum(self, current_epoch):
        if self.momentum_cosine:
            self.current_momentum = 1 - (1 - self.base_momentum) * (
                (math.cos(math.pi * current_epoch / self.total_epochs) + 1) / 2
            )
        else:
            self.current_momentum = self.base_momentum
    
    def _update_target_network(self):
        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = (
                self.current_momentum * param_k.data 
                + (1 - self.current_momentum) * param_q.data
            )
        
        for param_q, param_k in zip(
            self.projector_q.parameters(), self.projector_k.parameters()
        ):
            param_k.data = (
                self.current_momentum * param_k.data 
                + (1 - self.current_momentum) * param_q.data
            )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        view1 = batch["view1"]
        view2 = batch["view2"]
        gene_features = batch.get("gene_features", None)
        batch_matrix = batch.get("batch_matrix", None)
        
        kwargs = {"gene_features": gene_features} if gene_features is not None else {}
        
        embed_q1 = self.encoder(view1, **kwargs)
        proj_q1 = self.projector_q(embed_q1)
        pred_q1 = self.predictor_q(proj_q1)
        
        embed_q2 = self.encoder(view2, **kwargs)
        proj_q2 = self.projector_q(embed_q2)
        pred_q2 = self.predictor_q(proj_q2)
        
        with torch.no_grad():
            embed_k1 = self.encoder_k(view1, **kwargs)
            proj_k1 = self.projector_k(embed_k1)
            
            embed_k2 = self.encoder_k(view2, **kwargs)
            proj_k2 = self.projector_k(embed_k2)
        
        loss1 = self._moco_loss(pred_q1, proj_k2)
        loss2 = self._moco_loss(pred_q2, proj_k1)
        moco_loss = (loss1 + loss2) / 2
        
        loss = moco_loss
        losses_dict = {"moco_loss": moco_loss}
        
        if self.supervised_weight > 0 and batch_matrix is not None:
            sup_loss = self.supervised_loss_fn(embed_q1, batch_matrix)
            loss = loss + self.supervised_weight * sup_loss
            losses_dict["supervised_loss"] = sup_loss
            losses_dict["total_loss"] = loss
        
        return {"loss": loss, **losses_dict}
    
    def _moco_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        N = q.shape[0]
        logits = torch.mm(q, k.t()) / self.temperature
        
        if torch.distributed.is_initialized():
            labels = (torch.arange(N, dtype=torch.long).cuda() + 
                     N * torch.distributed.get_rank())
        else:
            labels = torch.arange(N, dtype=torch.long).cuda()
        
        loss = F.cross_entropy(logits, labels) * (2 * self.temperature)
        
        return loss
    
    def get_embedding(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encoder(x, **kwargs)



def create_ssl_model(
    method,
    encoder: nn.Module,
    embed_dim: int,
    **kwargs
) -> nn.Module:
    if method == 'byol':
        return BYOLModel(encoder, embed_dim, **kwargs)
    
    elif method == 'simclr':
        return SimCLRModel(encoder, embed_dim, **kwargs)
    
    elif method == 'wsl':
        return WSLModel(encoder, embed_dim, **kwargs)
    
    elif method == 'mae':
        return ViTMAE(encoder, **kwargs)
    
    elif method == 'mocov3':
        return MoCoV3Model(encoder, embed_dim, **kwargs)
    else:
        raise ValueError(f"unsupported method: {method}")


