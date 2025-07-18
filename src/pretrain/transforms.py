import torch
import torch.nn.functional as F
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class ResizeTransform:
    def __init__(self, size=(256, 256), mode="area"):
        self.size = size
        self.mode = mode

    def __call__(self, tensor):
        if tensor.ndim == 3:  # [C, H, W]
            tensor = tensor.unsqueeze(0)
            out = F.interpolate(tensor, size=self.size, mode=self.mode)
            return out.squeeze(0)
        elif tensor.ndim == 4:  # [N, C, H, W]
            return F.interpolate(tensor, size=self.size, mode=self.mode)
        else:
            raise ValueError("Unsupported tensor shape for resize.")


class CenterCropTransform:
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_h = self.crop_w = crop_size
        else:
            self.crop_h, self.crop_w = crop_size

    def __call__(self, tensor):
        # tensor: [N, C, H, W]
        n, c, h, w = tensor.shape
        th, tw = self.crop_h, self.crop_w
        i = (h - th) // 2
        j = (w - tw) // 2
        return tensor[:, :, i:i+th, j:j+tw]



class RandomCropTransform:
    def __init__(self, crop_size):
        if isinstance(crop_size, int):
            self.crop_h = self.crop_w = crop_size
        else:
            self.crop_h, self.crop_w = crop_size

    def __call__(self, tensor):
        if tensor.ndim == 4:
            # tensor: [N, C, H, W]
            n, c, h, w = tensor.shape
            th, tw = self.crop_h, self.crop_w
            if h < th or w < tw:
                raise ValueError("Crop size should be <= image size")
            top = torch.randint(0, h - th + 1, size=(n,))
            left = torch.randint(0, w - tw + 1, size=(n,))
            crops = []
            for idx in range(n):
                crop = tensor[idx, :, top[idx]:top[idx]+th, left[idx]:left[idx]+tw]
                crops.append(crop.unsqueeze(0))
            return torch.cat(crops, dim=0)
        elif tensor.ndim == 3:
            # tensor: [C, H, W]
            c, h, w = tensor.shape
            th, tw = self.crop_h, self.crop_w
            if h < th or w < tw:
                raise ValueError("Crop size should be <= image size")
            top = torch.randint(0, h - th + 1, size=(1,)).item()
            left = torch.randint(0, w - tw + 1, size=(1,)).item()
            return tensor[:, top:top+th, left:left+tw]
        else:
            raise ValueError("Input tensor must be 3D or 4D (got {}D)".format(tensor.ndim))
    
class RandomResizedCrop:
    def __init__(self, size, scale=(0.08, 1.0)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale

    def __call__(self, imgs):
        # imgs: [N, C, H, W]
        N, C, H, W = imgs.shape
        out = []
        for n in range(N):
            img = imgs[n]
            area = H * W
            success = False
            for _ in range(10):
                target_area = area * torch.empty(1).uniform_(*self.scale).item()
                aspect_ratio = torch.empty(1).uniform_(3. / 4, 4. / 3).item()
                new_w = int(round((target_area * aspect_ratio) ** 0.5))
                new_h = int(round((target_area / aspect_ratio) ** 0.5))
                if 0 < new_w <= W and 0 < new_h <= H:
                    top = torch.randint(0, H - new_h + 1, (1,)).item()
                    left = torch.randint(0, W - new_w + 1, (1,)).item()
                    crop = img[:, top:top + new_h, left:left + new_w]
                    crop = F.interpolate(crop.unsqueeze(0), size=self.size, mode="area", align_corners=False)
                    out.append(crop)
                    success = True
                    break
            if not success:
                i = (H - self.size[0]) // 2
                j = (W - self.size[1]) // 2
                crop = img[:, i:i+self.size[0], j:j+self.size[1]]
                crop = F.interpolate(crop.unsqueeze(0), size=self.size, mode="area", align_corners=False)
                out.append(crop)
        return torch.cat(out, dim=0)  # [N, C, h, w]
    
    
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        # imgs: [N, C, H, W]
        N = imgs.shape[0]
        flipped = []
        for n in range(N):
            if torch.rand(1).item() < self.p:
                flipped.append(torch.flip(imgs[n], dims=[2]).unsqueeze(0))
            else:
                flipped.append(imgs[n].unsqueeze(0))
        return torch.cat(flipped, dim=0)
    
    
class FluorescenceChannelAdjustment(object):
    def __init__(self, intensity_range=(0.8, 1.2), p=0.5):
        self.intensity_range = intensity_range
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            result = img.clone()
            
            for c in range(img.shape[0]):
                intensity_factor = random.uniform(self.intensity_range[0], self.intensity_range[1])
                result[c] = img[c] * intensity_factor
                
            return result
        return img
    
class BackgroundAutofluorescence(object):
    def __init__(self, intensity=(0.01, 0.1), p=0.3):
        self.intensity = intensity
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            result = img.clone()
            
            # generate global background autofluorescence
            background_level = random.uniform(self.intensity[0], self.intensity[1])
            background = torch.ones_like(img) * background_level
            
            for c in range(img.shape[0]):
                channel_factor = random.uniform(0.5, 1.5) 
                result[c] = img[c] + background[c] * channel_factor
                
            if result.max() <= 1:
                result = torch.clamp(result, 0, 1)
            else:
                result = torch.clamp(result, 0, 255)
                
            return result
        return img
    
class FluorescenceContrastAdjustment(object):
    def __init__(self, contrast_range=(0.8, 1.3), p=0.5):
        self.contrast_range = contrast_range
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            result = img.clone()
            
            for c in range(img.shape[0]):
                if random.random() < 0.5:  # randomly select a channel
                    contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
                    mean = torch.mean(img[c])
                    result[c] = (img[c] - mean) * contrast_factor + mean
                    
                    if result[c].max() <= 1:
                        result[c] = torch.clamp(result[c], 0, 1)
                    else:
                        result[c] = torch.clamp(result[c], 0, 255)
                    
            return result
        return img
    
    



