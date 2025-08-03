#!/usr/bin/env python3
"""
U-Net model for soil horizon depth regression
Predicts depth values in centimeters for each horizon boundary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import torchvision.transforms as transforms
from PIL import Image


class LightweightUNet(nn.Module):
    """Lightweight U-Net for horizon depth regression"""
    
    def __init__(self, max_horizons: int = 7):
        """
        Initialize U-Net model
        
        Args:
            max_horizons: Maximum number of horizon boundaries to predict
        """
        super().__init__()
        self.max_horizons = max_horizons
        
        # Encoder (downsampling path)
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Decoder (upsampling path)
        self.dec4 = self.conv_block(512 + 256, 256)
        self.dec3 = self.conv_block(256 + 128, 128)
        self.dec2 = self.conv_block(128 + 64, 64)
        self.dec1 = self.conv_block(64 + 32, 32)
        
        # Global feature extraction for depth regression
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.depth_regressor = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, max_horizons),
            nn.ReLU(inplace=True)  # Ensure positive depth values
        )
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
    def conv_block(self, in_channels: int, out_channels: int):
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """Forward pass"""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = F.interpolate(b, e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = F.interpolate(d4, e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Global pooling and depth regression
        global_features = self.global_pool(d1).flatten(1)
        depths = self.depth_regressor(global_features)
        
        return depths


class SoilHorizonDepthDataset(Dataset):
    """Dataset for soil horizon depth prediction"""
    
    def __init__(self, data_dir: str, transform=None, max_horizons: int = 7):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing processed images and metadata
            transform: Image transformations
            max_horizons: Maximum number of horizons to predict
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_horizons = max_horizons
        self.samples = []
        
        self._scan_data()
        
    def _scan_data(self):
        """Scan data directory for valid samples"""
        for metadata_file in self.data_dir.glob("*_metadata.json"):
            sample_name = metadata_file.name.replace("_metadata.json", "")
            processed_image = self.data_dir / f"{sample_name}_processed.png"
            
            if processed_image.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    horizon_info = metadata.get('horizon_info', {})
                    if horizon_info.get('has_horizon_labels', False):
                        # Extract depth values from horizon_depths_cm
                        horizon_depths = horizon_info['horizon_depths_cm']
                        if horizon_depths:
                            # Get the end depths (bottom boundaries of each horizon)
                            depth_values = [float(depth[1]) for depth in horizon_depths]
                            
                            self.samples.append({
                                'sample_name': sample_name,
                                'image_path': processed_image,
                                'depth_values': depth_values,
                                'num_horizons': len(depth_values),
                                'metadata': metadata
                            })
                            
                except Exception as e:
                    print(f"Erreur lors du chargement {metadata_file}: {e}")
        
        print(f"Trouvé {len(self.samples)} échantillons avec données de profondeur")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image_path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for transforms
        image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = to_tensor(image)
        
        # Prepare target depths
        depth_values = sample['depth_values']
        target = torch.zeros(self.max_horizons, dtype=torch.float32)
        
        # Fill in actual depth values
        for i, depth in enumerate(depth_values[:self.max_horizons]):
            target[i] = float(depth)
        
        # Create mask for valid horizons (non-zero depths)
        valid_mask = torch.zeros(self.max_horizons, dtype=torch.float32)
        valid_mask[:len(depth_values)] = 1.0
        
        return {
            'image': image,
            'depths': target,
            'valid_mask': valid_mask,
            'num_horizons': len(depth_values),
            'sample_name': sample['sample_name']
        }


def get_transforms(mode: str = 'train'):
    """Get data transforms with only brightness/contrast adjustments"""
    
    if mode == 'train':
        return transforms.Compose([
            ResizeByMaxSide(512),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Only lighting changes
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            ResizeByMaxSide(512),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class ResizeByMaxSide:
    """Resize image so that the longer side = max_side, keeping aspect ratio"""
    def __init__(self, max_side=512):
        self.max_side = max_side

    def __call__(self, img: Image.Image):
        w, h = img.size
        scale = self.max_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.BILINEAR)


class DepthRegressionLoss(nn.Module):
    """Custom loss function for depth regression"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        
    def forward(self, predictions, targets, valid_mask):
        """
        Calculate loss only for valid horizons
        
        Args:
            predictions: Predicted depths (B, max_horizons)
            targets: Target depths (B, max_horizons)
            valid_mask: Mask indicating valid horizons (B, max_horizons)
        """
        # Calculate MSE loss
        loss = self.mse(predictions, targets)
        
        # Apply mask to ignore invalid horizons
        masked_loss = loss * valid_mask
        
        # Calculate mean loss over valid horizons only
        if self.reduction == 'mean':
            total_loss = masked_loss.sum()
            total_valid = valid_mask.sum()
            return total_loss / (total_valid + 1e-8)
        else:
            return masked_loss


def create_data_loaders(data_dir: str, batch_size: int = 8, val_split: float = 0.2, 
                       num_workers: int = 0):
    """Create data loaders for training and validation"""
    
    # Create datasets
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    full_dataset = SoilHorizonDepthDataset(data_dir, transform=train_transform)
    
    if len(full_dataset) == 0:
        raise ValueError("Aucune donnée valide trouvée")
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Set validation transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, full_dataset


def calculate_metrics(predictions, targets, valid_mask):
    """Calculate regression metrics"""
    # Apply mask
    pred_valid = predictions * valid_mask
    target_valid = targets * valid_mask
    
    # Calculate MAE (Mean Absolute Error) only for valid horizons
    mae = torch.abs(pred_valid - target_valid) * valid_mask
    total_mae = mae.sum() / (valid_mask.sum() + 1e-8)
    
    # Calculate RMSE (Root Mean Square Error)
    mse = ((pred_valid - target_valid) ** 2) * valid_mask
    rmse = torch.sqrt(mse.sum() / (valid_mask.sum() + 1e-8))
    
    return {
        'mae': total_mae.item(),
        'rmse': rmse.item()
    }


if __name__ == "__main__":
    """Test the depth regression model"""
    
    print("=== Test du modèle de régression des profondeurs ===")
    
    # Test model
    model = LightweightUNet(max_horizons=7)
    print(f"Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test dataset
    data_dir = "data/processed"
    try:
        dataset = SoilHorizonDepthDataset(data_dir)
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Forme de l'échantillon: image {sample['image'].shape}")
            print(f"Profondeurs cibles: {sample['depths']}")
            print(f"Masque valide: {sample['valid_mask']}")
            print(f"Nombre d'horizons: {sample['num_horizons']}")
        else:
            print("Dataset vide")
    except Exception as e:
        print(f"Test du dataset échoué: {e}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
        print(f"Forme de sortie du modèle: {output.shape}")
        print(f"Exemple de prédictions: {output[0]}")