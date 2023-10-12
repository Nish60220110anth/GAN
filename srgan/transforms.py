import torch
import albumentations as A
import albumentations.pytorch as AP
from albumentations.pytorch import ToTensorV2

highres_transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
    AP.ToTensorV2()
])

lowres_transform = A.Compose([
    A.Resize(width=64, height=64),
    A.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
    AP.ToTensorV2()
])

both_transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
    AP.ToTensorV2()
])
