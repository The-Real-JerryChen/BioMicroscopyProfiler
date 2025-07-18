import os
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import argparse
from dataset import *
from clsmodel import *
from trainer import *
import random
import numpy as np

import sys
sys.path.append('..') 
from models.vit import *
from models.gr_loss import *

import logging

def main():
    parser = argparse.ArgumentParser(description='Evaluate CLS Model')
    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=36)

    parser.add_argument('--num_classes', type=int, default=247)
    parser.add_argument('--backbone', type=str, default='vit-s')
    parser.add_argument('--pooling_type', type=str, default='mean')
    parser.add_argument('--use_cls', type=bool, default=True)

    parser.add_argument('--method', type=str, default='wsl')
    parser.add_argument('--total_pretrain_epochs', type=int, default=400)
    parser.add_argument('--test_model', type=str, default='best', choices=['best', 'last'])
    
    parser.add_argument('--cell_line', type=str, default='U2OS', choices=['HRCE', 'U2OS', 'VERO'])
    parser.add_argument('--ge_token', type=int, default=8)
    parser.add_argument('--ge_token_dim', type=int, default=256)
    parser.add_argument('--pkl_path', type=str, default='../data/scvi.pkl')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    parser.add_argument('--use_wandb', action='store_true', default=True)
    parser.add_argument('--wandb_project_name', type=str, default='Downstream-Finetuning')
    parser.add_argument('--wandb_run_name', type=str, default='downstream-run')
    parser.add_argument('--log_path', type=str, default='./eval_results')
    parser.add_argument('--supervised_weight', type=float, default=0.0)
    parser.add_argument('--loss', type=str, default='con', choices=['con', 'lap'])
    
    img_root_path = 'Data/rxrx'

    args = parser.parse_args()
    if args.cell_line == 'U2OS':
        args.csv_path = '../data/u2os_data.csv'
        args.root_dir = os.path.join(img_root_path, 'rxrx1/images')
        args.num_classes = 1138
        args.dataset = 'rxrx1'
        
    elif args.cell_line == 'VERO':
        args.csv_path = '../data/vero_data.csv'
        args.root_dir = os.path.join(img_root_path, 'RxRx19a/RxRx19a/images')
        args.num_classes = 31
        args.dataset = 'rx19'
    else:
        args.csv_path = '../data/hrce_data.csv'
        args.root_dir = os.path.join(img_root_path, 'RxRx19a/RxRx19a/images')
        args.num_classes = 1512
        args.dataset = 'rx19'

    args.checkpoint_path = os.path.join(args.checkpoint_path, args.cell_line, args.method, str(args.ge_token), args.test_model, str(args.supervised_weight))
    os.makedirs(os.path.join(args.log_path,  args.cell_line, args.test_model, str(args.supervised_weight)), exist_ok=True)
    log_file = os.path.join(args.log_path, args.cell_line, args.test_model, str(args.supervised_weight), f'{args.loss}_{args.cell_line}_{args.method}_ge{args.ge_token}_dim{args.ge_token_dim}.log')
    logger = logging.getLogger('results_logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    logger.info(args)
    
    train_loader, val_loader, test_loader = create_data_loaders(args)
    vit_config = ViTConfig(
        model_type='small',  
        in_chans=6 if args.dataset == 'rxrx1' else 5,
        img_size=224,
        patch_size=16,
        use_cls_token=args.use_cls,
        pooling_type=args.pooling_type,
        init_pos_embed_type='learnable',
        ge_token=args.ge_token,
        ge_token_dim = args.ge_token_dim,
        drop_path_rate = 0.,
        pretrained=False,
    )
    encoder = ViTModel(vit_config)
    checkpoint_path = os.path.join('../pretrain/checkpoints', str(args.total_pretrain_epochs), args.method, str(args.ge_token), str(args.ge_token_dim))
    
    if args.dataset == 'rxrx1':
        if args.test_model == 'best':
            encoder = load_pretrained_weights(encoder, checkpoint_path, f'best_{args.method}_vit-s.pt')
        # else: 
        #     encoder.load_state_dict(torch.load(os.path.join(checkpoint_path, 'final_encoder_vit-s.pt'), map_location='cpu')['encoder_state_dict'], strict=True)
            
    else: 
        encoder = load_pretrained_weights(encoder, checkpoint_path, f'best_{args.method}_vit-s.pt')
            
    all_results = []
    all_seeds = [42, 229, 1234, 2025, 114514]
    for seed in all_seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        gr_loss = LaplacianRegularizationLoss(normalize_features=True, lambda_reg=1.0) if args.loss == 'lap' else GraphContrastiveLoss(normalize_features=True, lambda_reg=1.0, temperature=0.1)
        model = CLSModel(
            encoder=encoder,
            embed_dim = vit_config.embed_dim,
            num_classes=args.num_classes,
            use_cls_token=args.use_cls,
            pooling_type=args.pooling_type,
            supervised_loss_fn=gr_loss,
            supervised_weight=args.supervised_weight,
        )
        config = vars(args)
        trainer = CLSModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
        )
        trainer.train()
        loss, top1_acc, top5_acc, top10_acc = trainer.test()
        all_results.append({"loss": loss, "top1_acc": top1_acc, "top5_acc": top5_acc, "top10_acc": top10_acc})
    avg_loss = np.mean([result['loss'] for result in all_results])
    avg_top1 = np.mean([result['top1_acc'] for result in all_results])
    avg_top5 = np.mean([result['top5_acc'] for result in all_results])
    avg_top10 = np.mean([result['top10_acc'] for result in all_results])
    std_top1 = np.std([result['top1_acc'] for result in all_results])
    std_top5 = np.std([result['top5_acc'] for result in all_results])
    logger.info(f"Average loss: {avg_loss:.4f}, "
                f"Average Top-1 accuracy: {avg_top1:.4f}, \nAverage Top-5 accuracy: {avg_top5:.4f}, \nAverage Top-10 accuracy: {avg_top10:.4f}, "
                f"Standard deviation of Top-1 accuracy: {std_top1:.4f}, Standard deviation of Top-5 accuracy: {std_top5:.4f}")

if __name__ == '__main__':
    main()