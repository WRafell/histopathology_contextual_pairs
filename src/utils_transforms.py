from torchvision.transforms import v2
from PIL import Image
import torch


class Transforms:
    def __init__(self, resize_int: int = 224, use_prime_transforms: bool = True):
        self.use_prime_transforms = use_prime_transforms
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            # v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.8, 1.0), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomCrop(size=(128, 128)),
            v2.Resize(size=(resize_int, resize_int), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            # v2.RandomApply([v2.RandAugment()], p=0.2),
            # v2.RandomGrayscale(p=0.1),    
            # v2.RandomApply([v2.ElasticTransform(alpha=150.0)], p=0.2),
            # v2.RandomApply([v2.RandomEqualize()], p=0.2),
            # v2.RandomApply([v2.RandomChannelPermutation()], p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_transforms_prime = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.6, 1.0), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),            
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            v2.RandomApply([v2.RandAugment()], p=0.2),
            v2.RandomGrayscale(p=0.1),    
            v2.RandomApply([v2.RandomChannelPermutation()], p=0.2),
            v2.RandomApply([v2.ElasticTransform(alpha=150.0)], p=0.2),
            v2.RandomApply([v2.RandomEqualize()], p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def two_views_transform(self, x):
        v1 = self.train_transforms(x)
        if self.use_prime_transforms:
            v2 = self.train_transforms_prime(x)
        else:
            v2 = self.train_transforms(x)
        return v1, v2