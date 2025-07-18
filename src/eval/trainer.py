import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np
from typing import Union, List


def get_submatrix(full_matrix: pd.DataFrame, 
                  sirna_subset: Union[List[str], np.ndarray], 
                  ensure_order: bool = True) -> pd.DataFrame:
    if isinstance(sirna_subset, np.ndarray):
        sirna_subset = sirna_subset.tolist()
    
    submatrix = full_matrix.loc[sirna_subset, sirna_subset]
    
    if ensure_order:
        submatrix = submatrix.reindex(index=sirna_subset, columns=sirna_subset)
    
    return submatrix


full_matrix = pd.read_pickle("../data/u2os.pkl")
full_matrix[full_matrix < 0] = 0



class CLSModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        config: dict = None,
        local_rank: int = -1
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.total_steps = len(self.train_loader) * self.config.get('epochs', 1)
        self.checkpoint_path = self.config.get('checkpoint_path', 'checkpoints')

        opt_class = torch.optim.AdamW if self.config.get('optimizer', 'adamw').lower() == 'adamw' else torch.optim.Adam
        self.optimizer = opt_class(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=self.config.get('weight_decay', 0.0)
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['learning_rate'],
            total_steps=self.total_steps, 
            pct_start=self.config.get('pct_start', 0.1),
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=self.config.get('div_factor', 25.0),
            final_div_factor=self.config.get('final_div_factor', 1000.0),
        )

        self.use_wandb = self.config.get('use_wandb', True)

        wandb.init(
            project=self.config.get('wandb_project_name', 'Fine-tuning'),
            entity='mm4sci',
            name=self.config.get('wandb_run_name', f"{self.config.get('dataset', 'rxrx1')}-finetuning"),
            config=self.config
        )

        self.patience = self.config.get('early_stopping_patience', float('inf'))
        self.best_val_acc = 0.0
        self.early_stop_counter = 0

    def train_epoch(self, epoch: int) -> tuple:
        self.model.train()
        total_loss, total_acc = 0.0, 0.0
        train_iter = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')

        for batch_idx, batch in enumerate(train_iter):
            processed = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else
                             [t.to(self.device) for t in v] if isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v)
                             else v)
                         for k, v in batch.items()}
            processed["batch_matrix"] = get_submatrix(full_matrix, processed["sirna"])

            outputs = self.model.forward(processed)
            loss = outputs['loss']

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += outputs['acc'].item()

            train_iter.set_postfix(loss=loss.item(), lr=self.scheduler.get_last_lr()[0], acc=outputs['acc'].item())
            if self.use_wandb and batch_idx % 10 == 0:
                log = {'train_loss': loss.item(), 'train_acc': outputs['acc'].item(),
                       'learning_rate': self.scheduler.get_last_lr()[0], 'epoch': epoch, 'step': batch_idx}
                wandb.log(log)

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        if self.use_wandb:
            wandb.log({'epoch_train_loss': avg_loss, 'epoch_train_acc': avg_acc, 'epoch': epoch})
        return avg_loss, avg_acc

    @torch.no_grad()
    def validation_epoch(self, epoch: int) -> tuple:
        if self.val_loader is None:
            return None, None
        self.model.eval()
        total_loss, total_acc = 0.0, 0.0
        val_iter = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
        total_samples = 0

        for batch in val_iter:

            processed = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else
                    [t.to(self.device) for t in v] if isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v)
                    else v)
                for k, v in batch.items() if k != "batch_matrix"}
            
            outputs = self.model.forward(processed)
            loss = outputs['loss']
            labels = outputs['labels']
            bsz = labels.size(0)
            total_samples += bsz
            total_loss += loss.item() * bsz
            total_acc += outputs['acc'].item() * bsz
            val_iter.set_postfix(val_loss=loss.item(), val_acc=outputs['acc'].item())

        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        if self.use_wandb:
            wandb.log({'epoch_val_loss': avg_loss, 'epoch_val_acc': avg_acc, 'epoch': epoch})
        return avg_loss, avg_acc

    def train(self):
        for epoch in range(self.config['epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validation_epoch(epoch)

            if val_acc is not None and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
                self.save_checkpoint(f"best_model.pt", epoch, val_loss)
            else:
                self.early_stop_counter += 1

            print(f"Epoch {epoch+1}/{self.config['epochs']}, \
"                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, \
"                  f"Val Loss: {val_loss:.4f}, \
"                  f"Val Acc: {val_acc:.4f}, \
"                  f"Best Val Acc: {self.best_val_acc:.4f}, \
"                  f"Early Stop: {self.early_stop_counter}/{self.patience}")

            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    @torch.no_grad()
    def test(self, test_loader: DataLoader = None) -> tuple:
        best_model_path = os.path.join(self.checkpoint_path, "best_model.pt")
        self.model.load_state_dict(torch.load(best_model_path)['state_dict'])
        print(f"Loaded best model weights from {best_model_path}")
        loader = test_loader or self.test_loader
        assert loader is not None, "No test loader provided."
        self.model.eval()
        test_iter = tqdm(loader, desc='Test')
        total_loss = 0.0
        total_samples = 0
        total_top1 = 0
        total_top5 = 0
        total_top10 = 0
        
        for batch in test_iter:

            processed = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else
                    [t.to(self.device) for t in v] if isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v)
                    else v)
                for k, v in batch.items() if k != "batch_matrix"}
            outputs = self.model.forward(processed)
            logits = outputs['logits']
            labels = outputs.get('labels')
            bsz = labels.size(0)
            total_samples += bsz

            total_loss += outputs['loss'].item()* bsz

            topk_indices = torch.topk(logits, 10, dim=1).indices.cpu()

            total_top1  += (topk_indices[:, :1].eq(labels.unsqueeze(1))).any(1).sum().item()
            total_top5  += (topk_indices[:, :5].eq(labels.unsqueeze(1))).any(1).sum().item()
            total_top10 += topk_indices.eq(labels.unsqueeze(1)).any(dim=1).sum().item()
            
            test_iter.set_postfix(test_loss=outputs['loss'].item(), test_acc=outputs['acc'].item())


        avg_loss  = total_loss / total_samples
        top1_acc  = total_top1  / total_samples
        top5_acc  = total_top5  / total_samples
        top10_acc = total_top10 / total_samples

        print(f"Test Loss: {avg_loss:.4f}, "
              f"Top-1 Acc: {top1_acc:.4f}, Top-5 Acc: {top5_acc:.4f}, Top-10 Acc: {top10_acc:.4f}")
        return avg_loss, top1_acc, top5_acc, top10_acc

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        save_path = os.path.join(self.checkpoint_path, path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'loss': loss,
            'config': self.config
        }, save_path)
        print(f"Checkpoint saved to {save_path}")
