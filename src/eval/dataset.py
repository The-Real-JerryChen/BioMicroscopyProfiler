import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.io import read_image
import torch.distributed as dist
from torch.utils.data import DataLoader
import pickle
import numpy as np

class RXDataset(Dataset):
    def __init__(self, csv_path, root_dir, ge_representation_path, ge_dim = 128, dataset = 'rxrx1',
                 split='split1', mode='train', transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data[split] == mode].reset_index(drop=True)
        
        if dataset == 'rxrx1':
            self.label_col = 'sirna'
        else: 
            self.label_col = 'treatment'
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(self.data[self.label_col].unique()))}
        print('number of labels:', len(self.label_mapping))
        self.root_dir = root_dir
        self.transform = transform
        
        with open(ge_representation_path, 'rb') as f:
            self.ge_representation_dict = pickle.load(f)
        self.ge_dim = ge_dim
        
        self.dataset = dataset
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        if self.dataset == 'rxrx1':
            img_paths = [
                os.path.join(self.root_dir, row["experiment"], f"Plate{row['plate']}", f"{row['well']}_s{row['site']}_w{i}.png")
                for i in range(1, 7)
            ]
        else:
            img_paths = [
                os.path.join(self.root_dir, row["experiment"], f"Plate{row['plate']}", f"{row['well']}_s{row['site']}_w{i}.png")
                for i in range(1, 6)
            ]
        

        img_tensors = [read_image(p) for p in img_paths]  
        img = torch.cat(img_tensors, dim=0).float()
        
        if self.transform:
            img_tensor = self.transform(img)
        else: 
            img_tensor = img
        
        label = self.label_mapping[row[self.label_col]]
        cell_type = row["cell_type"]
        sirna = row[self.label_col]
        
        return {
            "pixels": img_tensor,
            "label": label,
            "sirna": sirna,
            "gene_features": self.ge_representation_dict[self.ge_dim][cell_type]
        }
        
        
def create_data_loaders(args):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=[-90, 90])
    ])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256,  antialias=True),
        transforms.CenterCrop(224)
    ])
    
    train_dataset = RXDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        ge_representation_path = args.pkl_path,
        ge_dim = args.ge_token_dim,
        dataset = args.dataset,
        mode='train',
        transform=train_transforms
    )
    
    val_dataset = RXDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        ge_representation_path = args.pkl_path,
        ge_dim = args.ge_token_dim,
        dataset = args.dataset,
        mode='valid',
        transform=valid_transforms
    )
    
    test_dataset = RXDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        ge_representation_path = args.pkl_path,
        ge_dim = args.ge_token_dim,
        dataset = args.dataset,
        mode='test',
        transform=valid_transforms
    )
    
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False  
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader