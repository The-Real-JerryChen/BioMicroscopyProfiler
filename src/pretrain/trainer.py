import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from enum import Enum
from typing import Dict, Optional
import pandas as pd 

# from utils import *

class SSLMethod(Enum):
    BYOL = 'byol'
    SimCLR = 'simclr'
    MAE = 'mae'
    WSL = 'wsl'
    MoCoV3 = 'mocov3'
    
class SSLTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: dict,
        local_rank: int = -1
    ):

        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.local_rank = local_rank
        self.is_main_process = local_rank in [-1, 0]
        self.use_ddp = local_rank != -1
        
        if self.use_ddp:
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self.model.to(self.device)
        if self.use_ddp:
            self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)
        
        self.total_steps = len(self.train_loader) * config['epochs']
        self.warmup_steps = config.get('warmup_steps', 0) * self.total_steps
        
        opt_class = AdamW if config.get('optimizer', 'adamw').lower() == 'adamw' else torch.optim.Adam
        self.optimizer = opt_class(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.0)
        )

        if self.warmup_steps > 0:
            self.warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=self.warmup_steps)
        else:
            self.warmup_scheduler = None
            
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - self.warmup_steps,
            eta_min=config.get('min_lr', 1e-6)
        )
        
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)  


        
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb and self.is_main_process:
            wandb.init(
                project=config.get('wandb_project_name', 'SSL-Training'),
                entity = 'mm4sci',
                name=config.get('wandb_run_name', f"{config.get('ssl_method', 'ssl')}-training"),
                config=config
            )
        
        self.ssl_method = config.get('ssl_method', 'byol')
        
        self.patience = config.get('early_stopping_patience', float('inf'))
    
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_ssl_loss = 0.0
        total_supervised_loss = 0.0
        
        if self.use_ddp:
            self.train_loader.sampler.set_epoch(epoch)
        
        train_iter = self.train_loader
        if self.is_main_process:
            train_iter = tqdm(train_iter, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(train_iter):
            processed_batch = {}
            sirna_set = batch['sirna']
            # batch_matrix = get_submatrix(full_matrix,sirna_set)
            # processed_batch['batch_matrix'] = batch_matrix

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    processed_batch[k] = v.to(self.device,non_blocking=True)
                elif isinstance(v, list) and all(isinstance(item, torch.Tensor) for item in v):
                    processed_batch[k] = [item.to(self.device) for item in v]
                else:
                    processed_batch[k] = v
            
            if self.use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    if self.use_ddp:
                        outputs = self.model.module.forward(processed_batch)
                    else:
                        outputs = self.model.forward(processed_batch)
                    
                    loss = outputs["loss"]
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.use_ddp:
                    outputs = self.model.module.forward(processed_batch)
                else:
                    outputs = self.model.forward(processed_batch)
                
                loss = outputs["loss"]
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


            
            if self.ssl_method == "byol":
                self.model.module._update_momentum(epoch)
                self.model.module._update_target_network()
                
            elif self.ssl_method == "mocov3":
                # if self.use_ddp:
                self.model.module._update_momentum(epoch)
                self.model.module._update_target_network()

            
            steps = batch_idx + epoch * len(self.train_loader)
            if steps < self.warmup_steps:
                if self.warmup_scheduler is not None:
                    self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            total_loss += loss.item()
            
            if "ssl_loss" in outputs:
                ssl_loss = outputs["ssl_loss"].item() if isinstance(outputs["ssl_loss"], torch.Tensor) else outputs["ssl_loss"]
                total_ssl_loss += ssl_loss
            elif self.ssl_method == "byol" and "byol_loss" in outputs:
                ssl_loss = outputs["byol_loss"].item() if isinstance(outputs["byol_loss"], torch.Tensor) else outputs["byol_loss"]
                total_ssl_loss += ssl_loss
            elif self.ssl_method == "simclr" and "simclr_loss" in outputs:
                ssl_loss = outputs["simclr_loss"].item() if isinstance(outputs["simclr_loss"], torch.Tensor) else outputs["simclr_loss"]
                total_ssl_loss += ssl_loss
            elif self.ssl_method == "wsl" and "wsl_loss" in outputs:
                ssl_loss = outputs["wsl_loss"].item() if isinstance(outputs["wsl_loss"], torch.Tensor) else outputs["wsl_loss"]
                total_ssl_loss += ssl_loss
            elif self.ssl_method == "mae" and "mae_loss" in outputs:
                ssl_loss = outputs["mae_loss"].item() if isinstance(outputs["mae_loss"], torch.Tensor) else outputs["mae_loss"]
                total_ssl_loss += ssl_loss
            elif self.ssl_method == "mocov3" and "moco_loss" in outputs:
                ssl_loss = outputs["moco_loss"].item() if isinstance(outputs["moco_loss"], torch.Tensor) else outputs["moco_loss"]
                total_ssl_loss += ssl_loss
            
            if "supervised_loss" in outputs:
                sup_loss = outputs["supervised_loss"].item() if isinstance(outputs["supervised_loss"], torch.Tensor) else outputs["supervised_loss"]
                total_supervised_loss += sup_loss
            
            if self.is_main_process:
                current_lr = self.scheduler.get_last_lr()[0]
                train_iter.set_postfix(
                    loss=loss.item(),
                    lr=current_lr
                )
                
                if self.use_wandb and batch_idx % 10 == 0:
                    log_dict = {
                        "train_loss": loss.item(),
                        "learning_rate": current_lr,
                        "epoch": epoch,
                        "step": batch_idx + epoch * len(self.train_loader)
                    }
                    
                    if "ssl_loss" in outputs or f"{self.ssl_method}_loss" in outputs:
                        log_dict["ssl_loss"] = ssl_loss
                    
                    if "supervised_loss" in outputs:
                        log_dict["supervised_loss"] = sup_loss
                    
                    wandb.log(log_dict)
        
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_ssl_loss = total_ssl_loss / num_batches
        avg_supervised_loss = total_supervised_loss / num_batches
        
        if self.use_ddp:
            world_size = dist.get_world_size()
            loss_tensor = torch.tensor(avg_loss).to(self.device)
            ssl_loss_tensor = torch.tensor(avg_ssl_loss).to(self.device)
            sup_loss_tensor = torch.tensor(avg_supervised_loss).to(self.device)
            
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(ssl_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(sup_loss_tensor, op=dist.ReduceOp.SUM)
            
            avg_loss = loss_tensor.item() / world_size
            avg_ssl_loss = ssl_loss_tensor.item() / world_size
            avg_supervised_loss = sup_loss_tensor.item() / world_size
        
        if self.is_main_process and self.use_wandb:
            wandb.log({
                "epoch_train_loss": avg_loss,
                "epoch_ssl_loss": avg_ssl_loss,
                "epoch_supervised_loss": avg_supervised_loss,
                "epoch": epoch
            })
        
        return avg_loss
    
    def train(self):
        best_train_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            
            if self.is_main_process:
                checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')

                
                method_name = self.config.get('ssl_method', 'ssl')
                
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                backbone_name = self.config.get('backbone', 'encoder')
                ge_token = self.config.get('ge_token', 0)
                
                if (epoch + 1) % self.config.get('save_every', 10) == 0:
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, f'{method_name}_getoken_{ge_token}_epoch{epoch+1}.pt'),
                        epoch,
                        train_loss
                    )
                
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    self.save_checkpoint(
                        os.path.join(checkpoint_dir, f'best_{method_name}_{backbone_name}.pt'),
                        epoch,
                        train_loss
                    )
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                
                print(
                    f'Epoch {epoch+1}/{self.config["epochs"]}, '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Best Loss: {best_train_loss:.4f}, '
                    f'Early Stop Counter: {early_stop_counter}/{self.patience}'
                )
                
                if early_stop_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        if self.is_main_process:
            self.save_encoder(os.path.join(checkpoint_dir, f'final_encoder_{backbone_name}.pt'))
        
        if self.use_ddp:
            dist.barrier()
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):

        state_dict = self.model.module.state_dict() if self.use_ddp else self.model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }, path)
        
        print(f"Checkpoint saved to {path}")
    
    def save_encoder(self, path: str):
        model = self.model.module if self.use_ddp else self.model
        encoder_state_dict = model.encoder.state_dict()
        torch.save({
            'encoder_state_dict': encoder_state_dict,
            'config': self.config
        }, path)
        
        print(f"Encoder saved to {path}")



