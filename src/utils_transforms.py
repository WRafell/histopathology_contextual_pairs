from torchvision.transforms import v2
from PIL import Image
import torch


class StandardTransforms:
    def __init__(self, resize_int: int):
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomCrop(size=(150, 150)),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.8, 1.0), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            v2.RandomApply([v2.RandAugment()], p=0.1),
            v2.RandomGrayscale(p=0.1),    
            v2.RandomApply([v2.ElasticTransform(alpha=150.0)], p=0.2),
            v2.RandomApply([v2.RandomEqualize()], p=0.2),
            v2.RandomApply([v2.RandomChannelPermutation()], p=0.2),
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


    def __call__(self, x):
        v1 = self.train_transforms(x)
        v2 = self.train_transforms_prime(x)
        return v1, v2
 

class ContextSamplingTransforms:
    def __init__(self, resize_int: int):
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            # v2.RandomCrop(size=(150, 150)),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.6, 1.0), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            v2.RandomGrayscale(p=0.2), 
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, x):
        v1 = self.train_transforms(x)
        v2 = self.train_transforms(x)            
        return v1, v2
    

class RegionTransforms:
    def __init__(self, resize_int: int):
        self.train_transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.CenterCrop(size=(resize_int, resize_int)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),            
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.train_transforms_prime = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resize_int, resize_int), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),            
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.5
            ),
            v2.RandomGrayscale(p=0.2),    
            v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __call__(self, x):
        v1 = self.train_transforms(x)
        v2 = self.train_transforms_prime(x)
        return v1, v2
    

class RandStainNATransforms:
    def __init__(self, resize_int: int, randstainna_augment: object):
        self.train_transforms = v2.Compose([
            randstainna_augment,
            v2.ToImage(),            
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomCrop(size=(150, 150)),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.8, 1.0), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
        ])

        self.train_transforms_prime = v2.Compose([
            randstainna_augment,
            v2.ToImage(),            
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomResizedCrop(size=(resize_int, resize_int), scale=(0.6, 1.0), interpolation=Image.BICUBIC, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),    
            v2.ToDtype(torch.float32, scale=True),            
        ])
        pass


    def __call__(self, x):
        v1 = self.train_transforms(x)
        v2 = self.train_transforms_prime(x)
        return v1, v2
    

# class DINOTransforms(object):
#     def __init__(
#             self, 
#             resize_int: int, 
#             global_crops_scale: tuple, 
#             local_crop_size: int, 
#             local_crops_scale: tuple, 
#             local_crops_number: int):
        
#         header = v2.Compose([
#             v2.ToImage(),
#             v2.ToDtype(torch.uint8, scale=True)])

#         flip_and_color_jitter = v2.Compose([
#             v2.RandomHorizontalFlip(p=0.5),
#             v2.RandomVerticalFlip(p=0.5),   
#             v2.RandomApply(
#                 [v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
#                 p=0.5
#             ),
#             v2.RandomGrayscale(p=0.2)])
        
#         normalize = v2.Compose([
#             v2.ToDtype(torch.float32, scale=True),
#             v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#         # first global crop
#         self.global_transfo1 = v2.Compose([
#             header,
#             v2.RandomResizedCrop(size=(resize_int, resize_int), scale=global_crops_scale, interpolation=Image.BICUBIC, antialias=True),
#             flip_and_color_jitter,
#             v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.7),
#             normalize,
#         ])
#         # second global crop
#         self.global_transfo2 = v2.Compose([
#             header,
#             v2.RandomResizedCrop(size=(resize_int, resize_int), scale=global_crops_scale, interpolation=Image.BICUBIC, antialias=True),
#             flip_and_color_jitter,
#             v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
#             normalize,
#         ])
#         # transformation for the local small crops
#         self.local_crops_number = local_crops_number
#         self.local_transfo = v2.Compose([
#             header,
#             v2.RandomResizedCrop(size=(local_crop_size, local_crop_size), scale=local_crops_scale, interpolation=Image.BICUBIC, antialias=True),
#             flip_and_color_jitter,
#             v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))], p=0.2),
#             normalize,
#         ])

#     def __call__(self, image):
#         global_views = []
#         global_views.append(self.global_transfo1(image))
#         global_views.append(self.global_transfo2(image))
#         local_views = []
#         for _ in range(self.local_crops_number):
#             local_views.append(self.local_transfo(image))
#         return global_views + local_views

    

