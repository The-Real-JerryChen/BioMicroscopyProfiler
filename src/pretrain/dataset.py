import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.io import read_image
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pickle
import numpy as np
from transforms import Compose, ResizeTransform, RandomCropTransform
from ssl_transforms import SSLTransforms

class RXDataset(Dataset):
    def __init__(self, csv_path, root_dir, ge_representation_path = None, ge_dim = 128, transform=None):
        self.data = pd.read_csv(csv_path)
        
        self.label_mapping = {label: idx for idx, label in enumerate(sorted(self.data["sirna"].unique()))}
        
        self.root_dir = root_dir
        self.transform = transform
        if ge_representation_path:
            with open(ge_representation_path, 'rb') as f:
                self.ge_representation_dict = pickle.load(f)
            self.ge_dim = ge_dim

            if ge_dim in self.ge_representation_dict.keys():
                self.domain_feature_dict = self.ge_representation_dict[ge_dim]
            elif str(ge_dim) in self.ge_representation_dict.keys():
                self.domain_feature_dict = self.ge_representation_dict[str(ge_dim)]
            else:
                raise ValueError(f"Domain feature dimension {ge_dim} not found in latent representation dict.")
        else:
            self.domain_feature_dict = None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        img_paths = [
            os.path.join(self.root_dir, row["experiment"], f"Plate{row['plate']}", f"{row['well']}_s{row['site']}_w{i}.png")
            for i in range(1, 7)
        ]
        
        # img_channels = [transforms.ToTensor()(Image.open(p)) for p in img_paths]
        # img_tensor = torch.cat(img_channels, dim=0)
        img_tensors = [read_image(p) for p in img_paths]  
        img = torch.cat(img_tensors, dim=0).float()
        
        if self.transform:
            img_tensor = self.transform(img)
        
        label = self.label_mapping[row["sirna"]]
        cell_type = row["cell_type"]
        sirna = row["sirna"]
        
        if cell_type in self.domain_feature_dict:
            domain_feature = self.domain_feature_dict[cell_type]
            if not isinstance(domain_feature, torch.Tensor):
                domain_feature = np.array(domain_feature, dtype=np.float32)
                domain_feature = torch.tensor(domain_feature, dtype=torch.float)
        else:
            raise ValueError(f"Cell type {cell_type} not found in latent representation dict.")
        
        return {
            "pixels": img_tensor,
            "label": label,
            "cell_type": cell_type,
            "sirna": sirna,
            "gene_features": domain_feature
        }


class SSLDatasetAdapter(Dataset):
    def __init__(self, base_dataset: RXDataset, ssl_method, image_size: int = 224):
        self.base_dataset = base_dataset
        self.ssl_method = ssl_method
        self.transform = SSLTransforms(method=ssl_method, image_size=image_size)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        image = sample["pixels"]
        gene_features = sample["gene_features"]
        label = sample["label"]
        sirna = sample['sirna']           
 

        if self.ssl_method == 'byol':
            view1, view2 = self.transform(image)
            return {
                "view1": view1,
                "view2": view2,
                "gene_features": gene_features,
                "labels": label,
                "sirna": sirna,
            }
        elif self.ssl_method == 'simclr':
            view1, view2 = self.transform(image)
            return {
                "view1": view1,
                "view2": view2,
                "gene_features": gene_features,
                "labels": label,
                "sirna": sirna,
            }
        
        elif self.ssl_method == 'wsl':
            transformed_img = self.transform(image)
            return {
                "imgs": transformed_img,
                "gene_features": gene_features,
                "labels": label,
                "sirna": sirna,
                # "original_img": image
            }
        
        elif self.ssl_method == 'mae':
            images = self.transform(image)
            return {
                "imgs": images,  # 
                "gene_features": gene_features,
                "labels": label,
                "sirna": sirna
            }

        elif self.ssl_method == 'mocov3':
            view1, view2 = self.transform(image)
            return {
                "view1": view1,
                "view2": view2,
                "gene_features": gene_features,
                "labels": label,
                "sirna": sirna,
            }

def create_data_loaders_ddp(args):
    train_transforms = Compose([
        ResizeTransform(size=(256, 256)),
    ])
    
    train_dataset = RXDataset(
        csv_path=args.csv_path,
        root_dir=args.root_dir,
        ge_representation_path = args.pkl_path,
        ge_dim = args.ge_token_dim,
        transform=train_transforms
    )
    print("train_dataset length: ", len(train_dataset))
    ssl_dataset = SSLDatasetAdapter(
        base_dataset=train_dataset,
        ssl_method=args.ssl_method,
        image_size=224
    )
    
    train_sampler = (
        DistributedSampler(train_dataset) 
        if dist.is_initialized() else None
    )
    
    train_loader = DataLoader(
        ssl_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,  
    )
    return train_loader
    
