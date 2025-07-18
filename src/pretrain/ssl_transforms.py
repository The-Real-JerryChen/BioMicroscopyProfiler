import torch
import torch.nn as nn
import torchvision.transforms as transforms
from enum import Enum
from typing import Dict, List, Tuple, Union
from transforms import FluorescenceChannelAdjustment, BackgroundAutofluorescence, FluorescenceContrastAdjustment


class SSLTransforms:
    def __init__(self, method, image_size: int = 224):
        self.method = method
        self.image_size = image_size
        
        base_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=[-90, 90])
        ]
        microscopy_jitter = [
            FluorescenceChannelAdjustment(p=0.5),
            BackgroundAutofluorescence(p=0.3),
            FluorescenceContrastAdjustment(p=0.5),
        ]
        if method == 'byol' or method == 'simclr':
            self.transform1 = transforms.Compose(base_transforms + microscopy_jitter + [
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], 
                                      p=1.0 if method == 'byol' else 0.5),
            ])
           
            self.transform2 = transforms.Compose(base_transforms + microscopy_jitter + [
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], 
                                      p=0.1 if method == 'byol' else 0.5),

            ])
            
        elif method == 'mocov3':
            color_jitter = transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            
            mocov3_transforms = base_transforms + microscopy_jitter + [
                # transforms.RandomApply([color_jitter], p=0.8),
                # transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
                # transforms.RandomSolarize(threshold=128, p=0.2),
            ]
            
            self.transform1 = transforms.Compose(mocov3_transforms)
            self.transform2 = transforms.Compose(mocov3_transforms)
    
        elif method == 'swav':
            swav_strong_aug = [
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
                # transforms.RandomGrayscale(p=0.2),
            ]
            
            self.transform1 = transforms.Compose(
                base_transforms + microscopy_jitter + swav_strong_aug
            )
            
            self.transform2 = transforms.Compose(
                base_transforms + microscopy_jitter + swav_strong_aug
            )
            

        elif method == 'dino':
            self.global_transform = transforms.Compose( [
                transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0), antialias=True),
                transforms.RandomHorizontalFlip()
            ] + microscopy_jitter +
            [
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            ])
            

            self.local_transform = transforms.Compose( [
                transforms.RandomResizedCrop(96, scale=(0.1, 0.4), antialias=True),
                transforms.RandomHorizontalFlip()
            ] + microscopy_jitter +
            [
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            ])
            
        elif method == 'mae':
            self.transform = transforms.Compose(base_transforms)
            
        elif method == 'wsl':
            self.transform = transforms.Compose(base_transforms + microscopy_jitter + [
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)],  p=0.2),
            ])
    
    def __call__(self, x: torch.Tensor) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, List[torch.Tensor]]]:

        if self.method in ['byol', 'simclr', 'swav', 'mocov3']:
            img1 = self.transform1(x)
            img2 = self.transform2(x)
            return img1, img2
        
        elif self.method == 'dino':
            global_views = [self.global_transform(x) for _ in range(2)]
            local_views = [self.local_transform(x) for _ in range(4)]
            
            return {
                'global_views': global_views,
                'local_views': local_views
            }
        elif self.method == 'mae':
            return self.transform(x)
            
        elif self.method == 'wsl':
            return self.transform(x)

