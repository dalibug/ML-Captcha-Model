"""
Data loader for CAPTCHA images with preprocessing and augmentation.
"""
import os
import re
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class CaptchaDataset(Dataset):
    """Dataset class for CAPTCHA images."""
    
    # All possible characters in CAPTCHA (alphanumeric)
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    NUM_CHARS = len(CHARS)
    CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
    IDX_TO_CHAR = {idx: char for idx, char in enumerate(CHARS)}
    SEQ_LENGTH = 5  # 5-character CAPTCHA
    
    def __init__(self, data_dir, transform=None, augment=False):
        """
        Args:
            data_dir: Directory containing CAPTCHA images
            transform: Optional transform to be applied on images
            augment: Whether to apply data augmentation
        """
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        
        # Get all image files
        self.image_files = []
        for file in os.listdir(data_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Extract label from filename (remove extension)
                label = os.path.splitext(file)[0]
                if len(label) == self.SEQ_LENGTH:
                    self.image_files.append(file)
        
        print(f"Found {len(self.image_files)} CAPTCHA images")
        
        # Create augmentation transforms if needed
        if self.augment and self.transform is None:
            self.transform = self._get_augmentation_transforms()
        elif self.transform is None:
            self.transform = self._get_base_transforms()
    
    def _get_base_transforms(self):
        """Base transforms: grayscale, resize, normalize."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((50, 200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    
    def _get_augmentation_transforms(self):
        """Transforms with data augmentation."""
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((50, 200)),
            # Data augmentation
            transforms.RandomRotation(degrees=5),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
            # Add noise
            transforms.Lambda(lambda x: self._add_noise(x))
        ])
    
    def _add_noise(self, tensor):
        """Add random noise to tensor."""
        noise = torch.randn_like(tensor) * 0.05
        return torch.clamp(tensor + noise, -1, 1)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Extract label from filename
        label_str = os.path.splitext(self.image_files[idx])[0]
        
        # Convert label to tensor (5 characters)
        label = torch.zeros(self.SEQ_LENGTH, dtype=torch.long)
        for i, char in enumerate(label_str):
            if char in self.CHAR_TO_IDX:
                label[i] = self.CHAR_TO_IDX[char]
            else:
                # Handle unknown characters
                label[i] = 0
        
        return image, label, label_str
    
    @staticmethod
    def decode_label(label_tensor):
        """Convert label tensor back to string."""
        if isinstance(label_tensor, torch.Tensor):
            label_tensor = label_tensor.cpu().numpy()
        return ''.join([CaptchaDataset.IDX_TO_CHAR[int(idx)] for idx in label_tensor])


class SubsetDataset(CaptchaDataset):
    """Subset dataset for train/val split."""
    def __init__(self, data_dir, image_files, transform=None, augment=False):
        self.data_dir = data_dir
        self.image_files = image_files
        self.transform = transform
        self.augment = augment
        
        if self.augment and self.transform is None:
            self.transform = self._get_augmentation_transforms()
        elif self.transform is None:
            self.transform = self._get_base_transforms()
    
    def __len__(self):
        return len(self.image_files)


def create_dataloaders(data_dir, train_split=0.8, batch_size=32, augment_train=True, num_workers=0):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Directory containing CAPTCHA images
        train_split: Fraction of data to use for training
        batch_size: Batch size for dataloaders
        augment_train: Whether to augment training data
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader
    """
    # Get all image files first
    image_files = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            label = os.path.splitext(file)[0]
            if len(label) == CaptchaDataset.SEQ_LENGTH:
                image_files.append(file)
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(image_files)
    
    train_size = int(train_split * len(image_files))
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]
    
    # Create datasets with proper file lists
    train_dataset = SubsetDataset(data_dir, train_files, augment=augment_train)
    val_dataset = SubsetDataset(data_dir, val_files, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

