import os
import socket

import argparse
import yaml
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from dataset import *
from trainer import SSLTrainer, SSLMethod

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ssl_models import *
from models.mae import *
from models.vit import *



def setup_ddp(rank, world_size):
    port = random.randint(10000, 30000)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port) 
    
    dist.init_process_group(
        backend='nccl',  
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_process(rank, world_size, args):
    setup_ddp(rank, world_size)
    
    set_seed(args.seed + rank)  
    
    config = vars(args)
    
    train_loader = create_data_loaders_ddp(args)
    

         
    if config['backbone'] == 'vit-s':
        print('use vit-s')
        vit_config = ViTConfig(
            model_type='small',  
            in_chans=6,
            img_size=224,
            patch_size=16,
            use_cls_token=config['use_cls'],
            pooling_type=config['pooling_type'],
            init_pos_embed_type='learnable',
            ge_token =  config['ge_token'] ,
            ge_token_dim = config['ge_token_dim'],
            drop_path_rate =  0.,
        )
        encoder = ViTModel(vit_config)

        

    model = create_ssl_model(
        method=args.ssl_method,
        encoder=encoder,
        embed_dim=vit_config.embed_dim,
        total_epochs=args.epochs,
        # supervised_loss_fn = LaplacianRegularizationLoss(),
        # supervised_weight = config['lp_reg']
    )

    trainer = SSLTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        local_rank=rank
    )
    
    trainer.train()
    
    cleanup()

def main():
    parser = argparse.ArgumentParser(description='ssl')

    parser.add_argument('--ssl_method', type=str, default='wsl', choices=['byol', 'simclr', 'mae', 'wsl', 'mocov3'])
    parser.add_argument('--batch_size', type=int, default=200, help='per GPU batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='adamw')

    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--save_every', type=int, default=40, help='save every n epochs')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=32, help='number of workers for data loading')
    parser.add_argument('--num_gpus', type=int, default=-1, help='number of GPUs, -1 means using all available GPUs')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--verbose', action='store_true', help='whether to print detailed information')
    parser.add_argument('--wandb_project_name', type=str, default='SSL-Training')
    parser.add_argument('--backbone', type=str, default='vit-s')
    parser.add_argument('--pooling_type', type=str, default='mean')
    parser.add_argument('--use_cls', action='store_true', default=True)
    parser.add_argument('--ge_token', type=int, default=0)
    parser.add_argument('--ge_token_dim', type=int, default=256)
    parser.add_argument('--lp_reg', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type = float, default = 0.05)
    parser.add_argument('--use_amp', action='store_true')#, default=False)

    # customize the following paths
    parser.add_argument('--csv_path', type=str, default='/data/rxrxmeta.csv')
    parser.add_argument('--root_dir', type=str, default='Data/rxrx/rxrx1/images')
    parser.add_argument('--pkl_path', type=str, default='data/scvi.pkl')
    
    
    
    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, str(args.epochs), args.ssl_method, str(args.ge_token), str(args.ge_token_dim))
    if args.ssl_method  == 'wsl':
        args.batch_size = 400
        args.weight_decay = 0.05
    elif args.ssl_method == 'simclr':
        args.batch_size = 400
        args.weight_decay = 1e-6
    elif args.ssl_method == 'mae':
        args.batch_size = 400
        args.weight_decay = 0.05
    elif args.ssl_method == 'byol':
        args.batch_size = 400
        args.weight_decay = 0.1
    elif args.ssl_method == 'mocov3':
        args.batch_size = 400
        args.weight_decay = 0.1
        args.learning_rate = 1e-4 * args.batch_size / 256
        
        
    if args.num_gpus == -1:
        args.num_gpus = torch.cuda.device_count()
    
    if args.num_gpus < 1:
        print("Error: No available GPUs! At least 1 GPU is required for DDP training.")
        return
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(args)
    
    print(f"Starting DDP training with {args.num_gpus} GPUs")
    print(f"SSL method: {args.ssl_method}")
    print(f"GE token: {args.ge_token}")
    print(f"GE token dim: {args.ge_token_dim}")
    print(f"Batch size: {args.batch_size} (per GPU)")
    print(f"Total batch size: {args.batch_size * args.num_gpus}")
    
    mp.spawn(
        train_process,
        args=(args.num_gpus, args),
        nprocs=args.num_gpus,
        join=True
    )

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()