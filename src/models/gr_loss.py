import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from typing import Union, List
import numpy as np


def deduplicate_interaction_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    matrix_deduped_rows = matrix.groupby(matrix.index).mean()
    matrix_deduped = matrix_deduped_rows.T.groupby(level=0).mean().T
    return matrix_deduped
    

class GraphContrastiveLoss(nn.Module):
    def __init__(self, normalize_features: bool = True, lambda_reg: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.normalize_features = normalize_features
        self.lambda_reg = lambda_reg
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, 
                interaction_matrix: Union[pd.DataFrame, torch.Tensor],
                sirna_labels: Union[List[str], torch.Tensor]) -> torch.Tensor:
        device = features.device

        unique_sirnas, inverse_indices = np.unique(sirna_labels, return_inverse=True)
        inverse_indices = torch.tensor(inverse_indices, device=device)
        n_unique = len(unique_sirnas)
        
        if isinstance(interaction_matrix, pd.DataFrame):
            interaction_matrix = deduplicate_interaction_matrix(interaction_matrix)
            unique_interaction = interaction_matrix.loc[unique_sirnas, unique_sirnas]
            interaction_values = unique_interaction.values
            interaction_matrix = torch.tensor(interaction_values, dtype=torch.float32, device=device)
        else:
            interaction_matrix = interaction_matrix.to(device)
        
        ones = torch.ones(features.size(0), 1, device=device)
        count_per_label = torch.zeros(n_unique, 1, device=device)
        count_per_label.scatter_add_(0, inverse_indices.unsqueeze(1), ones)
        
        avg_features = torch.zeros(n_unique, features.size(1), device=device)
        avg_features.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, features.size(1)), features)
        avg_features = avg_features / (count_per_label + 1e-8)
        
        if self.normalize_features:
            avg_features = F.normalize(avg_features, p=2, dim=1)
        
        sim = torch.matmul(avg_features, avg_features.t()) / self.temperature
        
        if interaction_matrix.max() > 1:
            W = interaction_matrix / 1000.0
        else:
            W = interaction_matrix.clone()
        
        mask = torch.eye(n_unique, device=device)
        sim = sim * (1 - mask)
        W = W * (1 - mask)
        
        log_softmax = F.log_softmax(sim, dim=1)
        loss = -(W * log_softmax).sum(dim=1).mean()
        
        return self.lambda_reg * loss
    
    
    
    
def get_submatrix(full_matrix: pd.DataFrame, 
                  sirna_subset: Union[List[str], np.ndarray], 
                  ensure_order: bool = True) -> pd.DataFrame:
    if isinstance(sirna_subset, np.ndarray):
        sirna_subset = sirna_subset.tolist()
    
    submatrix = full_matrix.loc[sirna_subset, sirna_subset]
    
    if ensure_order:
        submatrix = submatrix.reindex(index=sirna_subset, columns=sirna_subset)
    
    return submatrix




class LaplacianRegularizationLoss(nn.Module):
    def __init__(self, normalize_features: bool = True, lambda_reg: float = 1.0):
        super().__init__()
        self.normalize_features = normalize_features
        self.lambda_reg = lambda_reg
    
    def forward(self, features: torch.Tensor, 
                interaction_matrix: Union[pd.DataFrame, torch.Tensor],
                sirna_labels: Union[List[str], torch.Tensor] = None) -> torch.Tensor:
        device = features.device
        
        if sirna_labels is not None:
            unique_sirnas, inverse_indices = np.unique(sirna_labels, return_inverse=True)
            inverse_indices = torch.tensor(inverse_indices, device=device)
            n_unique = len(unique_sirnas)
            
            ones = torch.ones(features.size(0), 1, device=device)
            count_per_label = torch.zeros(n_unique, 1, device=device)
            count_per_label.scatter_add_(0, inverse_indices.unsqueeze(1), ones)
            
            avg_features = torch.zeros(n_unique, features.size(1), device=device)
            avg_features.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, features.size(1)), features)
            features = avg_features / (count_per_label + 1e-8)
            
            if isinstance(interaction_matrix, pd.DataFrame):
                interaction_matrix = deduplicate_interaction_matrix(interaction_matrix)
                unique_interaction = interaction_matrix.loc[unique_sirnas, unique_sirnas]
                interaction_matrix = torch.tensor(unique_interaction.values, dtype=torch.float32, device=device)
        else:
            if isinstance(interaction_matrix, pd.DataFrame):
                interaction_matrix = torch.tensor(interaction_matrix.values, dtype=torch.float32, device=device)
            else:
                interaction_matrix = interaction_matrix.to(device)
        
        if self.normalize_features:
            features = F.normalize(features, p=2, dim=1)
        
        batch_size = features.size(0)
        
        if interaction_matrix.max() > 1:
            W = interaction_matrix / 1000.0
        else:
            W = interaction_matrix.clone()
        
        W = W.fill_diagonal_(0.0)
        
        D = torch.diag(torch.sum(W, dim=1))
        
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diagonal(D) + 1e-6))
        L_norm = torch.eye(batch_size, device=device) - torch.mm(torch.mm(D_inv_sqrt, W), D_inv_sqrt)
        
        loss = torch.trace(torch.matmul(torch.matmul(features.t(), L_norm), features))
        
        return self.lambda_reg * loss