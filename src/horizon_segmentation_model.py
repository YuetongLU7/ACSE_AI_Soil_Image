#!/usr/bin/env python3
"""
DeepLabV3 model for soil horizon segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import random
from PIL import Image


class HorizonSegmentationModel(nn.Module):
    """DeepLabV3 model for soil horizon segmentation"""
    
    def __init__(self, num_classes: int = 8, backbone: str = 'resnet50', pretrained: bool = True):
        """
        Initialize the segmentation model
        
        Args:
            num_classes: Number of classes (background + up to 7 horizons)
            backbone: Backbone network ('resnet50' or 'resnet101')
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Select backbone network
        if backbone == 'resnet50':
            self.model = deeplabv3_resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.model = deeplabv3_resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier head
        self.model.classifier = DeepLabHead(2048, num_classes)
        
        # Auxiliary classifier (if exists)
        if hasattr(self.model, 'aux_classifier'):
            self.model.aux_classifier = DeepLabHead(1024, num_classes)
    
    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    
    def predict(self, x):
        """Prediction in inference mode"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            if isinstance(output, dict):
                output = output['out']
            return F.softmax(output, dim=1)


class SoilHorizonDataset(Dataset):
    """Soil horizon segmentation dataset"""
    
    def __init__(self, data_dir: str, transform=None, max_horizons: int = 7):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing processed data
            transform: Data transformations
            max_horizons: Maximum number of horizons
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_horizons = max_horizons
        self.samples = []
        
        # Scan data files
        self._scan_data()
        
    def _scan_data(self):
        """Scan data directory for valid samples"""
        for metadata_file in self.data_dir.glob("*_metadata.json"):
            # Extract sample name
            sample_name = metadata_file.name.replace("_metadata.json", "")
            
            # Check if required files exist
            processed_image = self.data_dir / f"{sample_name}_processed.png"
            horizon_mask = self.data_dir / f"{sample_name}_horizon_mask.png"
            
            if processed_image.exists() and horizon_mask.exists():
                # Read metadata to check if horizon labels exist
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('horizon_info', {}).get('has_horizon_labels', False):
                        self.samples.append({
                            'sample_name': sample_name,
                            'image_path': processed_image,
                            'mask_path': horizon_mask,
                            'metadata': metadata
                        })
                except Exception as e:
                    print(f"Ignorer fichier metadata invalide {metadata_file}: {e}")
        
        print(f"Trouvé {len(self.samples)} échantillons valides")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Read image
        image = cv2.imread(str(sample['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask = cv2.imread(str(sample['mask_path']), cv2.IMREAD_GRAYSCALE)
        
        # Limit number of horizons
        mask = np.clip(mask, 0, self.max_horizons)
        
        # Apply transforms
        if self.transform:
            # Transform image and mask together (keep synchronized)
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Default transform to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'sample_name': sample['sample_name'],
            'num_horizons': sample['metadata'].get('horizon_info', {}).get('num_horizons', 0)
        }


def get_transforms(mode: str = 'train', image_size: Tuple[int, int] = (512, 512)):
    """Get data transforms"""
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        if mode == 'train':
            return A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    except ImportError:
        print("Avertissement: albumentations non installé, utilisation des transformations de base")
        # Use basic torchvision transforms
        if mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


class SegmentationLoss(nn.Module):
    """Segmentation loss function"""
    
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.dice_loss = DiceLoss()
        
    def forward(self, predictions, targets):
        """
        Calculate loss
        
        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
        """
        if isinstance(predictions, dict):
            # DeepLabV3 returns dict
            main_loss = self.cross_entropy(predictions['out'], targets)
            dice_loss = self.dice_loss(predictions['out'], targets)
            
            total_loss = main_loss + 0.5 * dice_loss
            
            # Auxiliary loss
            if 'aux' in predictions:
                aux_loss = self.cross_entropy(predictions['aux'], targets)
                total_loss += 0.4 * aux_loss
                
            return total_loss
        else:
            main_loss = self.cross_entropy(predictions, targets)
            dice_loss = self.dice_loss(predictions, targets)
            return main_loss + 0.5 * dice_loss


class DiceLoss(nn.Module):
    """Dice loss function"""
    
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        """
        Calculate Dice loss
        
        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Ground truth labels (B, H, W)
        """
        # Convert predictions to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert labels to one-hot encoding
        num_classes = predictions.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss


def calculate_class_weights(dataset: SoilHorizonDataset, num_classes: int) -> torch.Tensor:
    """Calculate class weights for handling imbalanced data"""
    class_counts = torch.zeros(num_classes)
    
    print("Calcul des poids de classes...")
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample['mask']
        
        # Count pixels for each class
        unique, counts = torch.unique(mask, return_counts=True)
        for class_id, count in zip(unique, counts):
            if class_id < num_classes:
                class_counts[class_id] += count
    
    # Calculate weights (inverse proportion)
    total_pixels = class_counts.sum()
    class_weights = total_pixels / (num_classes * class_counts + 1e-8)
    
    print(f"Distribution des classes: {class_counts}")
    print(f"Poids des classes: {class_weights}")
    
    return class_weights


def create_data_loaders(data_dir: str, batch_size: int = 8, val_split: float = 0.2, 
                       image_size: Tuple[int, int] = (512, 512), num_workers: int = 4):
    """Create data loaders"""
    
    # Create dataset
    full_dataset = SoilHorizonDataset(
        data_dir=data_dir,
        transform=get_transforms('train', image_size)
    )
    
    if len(full_dataset) == 0:
        raise ValueError("Aucune donnée d'entraînement valide trouvée")
    
    # Split into train and validation sets
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Set different transforms for validation set
    val_transform = get_transforms('val', image_size)
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset


if __name__ == "__main__":
    """Test model and dataset"""
    
    print("=== Test du modèle de segmentation des horizons ===")
    
    # Test model
    model = HorizonSegmentationModel(num_classes=8, backbone='resnet50')
    print(f"Nombre de paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test dataset
    data_dir = "data/processed"
    try:
        dataset = SoilHorizonDataset(data_dir)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Forme de l'échantillon: image {sample['image'].shape}, masque {sample['mask'].shape}")
            print(f"Valeurs uniques dans le masque: {torch.unique(sample['mask'])}")
        else:
            print("Dataset vide")
    except Exception as e:
        print(f"Test du dataset échoué: {e}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
        if isinstance(output, dict):
            print(f"Forme de sortie du modèle: {output['out'].shape}")
        else:
            print(f"Forme de sortie du modèle: {output.shape}")