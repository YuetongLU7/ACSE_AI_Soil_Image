#!/usr/bin/env python3
"""
Element Grossiere Detection Model
Modèle UNet pour la détection des éléments grossiers du sol
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


class ElementGrossiereUNet(nn.Module):
    """UNet pour la segmentation des éléments grossiers"""
    
    def __init__(self, num_classes: int = 3):
        """
        Initialize UNet model
        
        Args:
            num_classes: Nombre de classes (arriere_plan, pierres, racines)
        """
        super().__init__()
        self.num_classes = num_classes
        
        # Encodeur (downsampling path)
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Décodeur (upsampling path)
        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        
        # Couche de classification finale
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Pooling et upsampling
        self.pool = nn.MaxPool2d(2, 2)
        
    def conv_block(self, in_channels: int, out_channels: int):
        """Créer un bloc de convolution"""
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
        # Encodeur avec skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Décodeur avec skip connections
        d4 = F.interpolate(b, e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = F.interpolate(d4, e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Classification finale
        output = self.final_conv(d1)
        
        return output


class ElementGrossiereDataset(Dataset):
    """Dataset pour la segmentation des éléments grossiers"""
    
    def __init__(self, dataset_dir: str, split: str = 'train', transform=None):
        """
        Initialize dataset
        
        Args:
            dataset_dir: Répertoire du dataset
            split: 'train', 'val', ou 'test'
            transform: Transformations d'augmentation
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.transform = transform
        
        # Charger les informations du dataset
        with open(self.dataset_dir / "dataset_info.json", 'r', encoding='utf-8') as f:
            self.dataset_info = json.load(f)
        
        self.classes = self.dataset_info['classes']
        self.images = self.dataset_info['images']
        
        # Diviser le dataset
        self.split_dataset()
        
    def split_dataset(self):
        """Diviser le dataset en train/val/test"""
        total_images = len(self.images)
        
        # Pour un petit dataset, utiliser toutes les images pour train et val
        if total_images <= 3:
            if self.split == 'train':
                self.images = self.images  # Utiliser toutes les images
            elif self.split == 'val':
                self.images = self.images  # Utiliser les mêmes images pour validation
            elif self.split == 'test':
                self.images = self.images[:min(1, total_images)]
        else:
            # 70% train, 20% val, 10% test
            train_end = int(0.7 * total_images)
            val_end = int(0.9 * total_images)
            
            if self.split == 'train':
                self.images = self.images[:train_end]
            elif self.split == 'val':
                self.images = self.images[train_end:val_end]
            elif self.split == 'test':
                self.images = self.images[val_end:]
        
        print(f"Split {self.split}: {len(self.images)} images")
    
    def load_mask_from_json(self, json_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
        """Charger le masque depuis un fichier JSON labelme"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Créer un masque vide
        height, width = image_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Parcourir toutes les formes
        for shape in data['shapes']:
            label = int(shape['label'])  # Convertir le label en entier
            points = np.array(shape['points'], dtype=np.int32)
            
            # Remplir le polygone avec le label
            cv2.fillPoly(mask, [points], label)
        
        return mask
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_info = self.images[idx]
        
        # Charger l'image
        image_path = self.dataset_dir / image_info['image_path']
        image = cv2.imread(str(image_path))
        
        if image is None:
            print(f"Erreur: Impossible de charger l'image: {image_path}")
            print(f"Le fichier existe-t-il? {image_path.exists()}")
            # Essayer avec un autre encodage ou ignorer cette image
            raise ValueError(f"Cannot load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Charger le masque depuis JSON
        annotation_path = self.dataset_dir / image_info['annotation_path']
        mask = self.load_mask_from_json(annotation_path, image.shape[:2])
        
        # Convertir en PIL pour les transformations
        image_pil = Image.fromarray(image)
        mask_pil = Image.fromarray(mask)
        
        # Appliquer les transformations
        if self.transform:
            # Appliquer la même transformation à l'image et au masque
            seed = np.random.randint(2147483647)
            
            np.random.seed(seed)
            torch.manual_seed(seed)
            image_pil = self.transform(image_pil)
            
            np.random.seed(seed)
            torch.manual_seed(seed)
            mask_pil = transforms.Compose([
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])(mask_pil)
        else:
            # Transformations par défaut
            default_transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image_pil = default_transform(image_pil)
            
            mask_transform = transforms.Compose([
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])
            mask_pil = mask_transform(mask_pil)
        
        # Convertir le masque en long tensor pour CrossEntropyLoss
        mask_pil = (mask_pil * 255).long().squeeze(0)
        
        return {
            'image': image_pil,
            'mask': mask_pil,
            'filename': Path(image_info['image_path']).name
        }


def get_transforms(mode: str = 'train'):
    """Obtenir les transformations d'augmentation"""
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


class WeightedCrossEntropyLoss(nn.Module):
    """Cross Entropy Loss avec poids pour gérer le déséquilibre des classes"""
    
    def __init__(self, class_weights=None):
        super().__init__()
        self.class_weights = class_weights
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
    
    def forward(self, predictions, targets):
        if self.class_weights is not None and predictions.is_cuda:
            self.class_weights = self.class_weights.cuda()
        
        return F.cross_entropy(predictions, targets, weight=self.class_weights)


def load_mask_from_json_static(json_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
    """Version statique pour charger le masque depuis un fichier JSON labelme"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Créer un masque vide
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Parcourir toutes les formes
    for shape in data['shapes']:
        label = int(shape['label'])  # Convertir le label en entier
        points = np.array(shape['points'], dtype=np.int32)
        
        # Remplir le polygone avec le label
        cv2.fillPoly(mask, [points], label)
    
    return mask


def calculate_class_weights(dataset_dir: str) -> List[float]:
    """Calculer les poids des classes pour gérer le déséquilibre"""
    dataset_path = Path(dataset_dir)
    
    with open(dataset_path / "dataset_info.json", 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    # Compter les pixels de chaque classe
    class_counts = {i: 0 for i in range(len(dataset_info['classes']))}
    
    for image_info in dataset_info['images']:
        # Charger l'image pour obtenir les dimensions
        image_path = dataset_path / image_info['image_path']
        image = cv2.imread(str(image_path))
        if image is None:
            continue
            
        # Générer le masque depuis le JSON
        annotation_path = dataset_path / image_info['annotation_path']
        mask = load_mask_from_json_static(annotation_path, image.shape[:2])
        
        if mask is not None:
            unique, counts = np.unique(mask, return_counts=True)
            for class_id, count in zip(unique, counts):
                if class_id < len(class_counts):
                    class_counts[class_id] += count
    
    # Calculer les poids inversement proportionnels à la fréquence
    total_pixels = sum(class_counts.values())
    class_weights = []
    
    for class_id in range(len(dataset_info['classes'])):
        if class_counts[class_id] > 0:
            weight = total_pixels / (len(dataset_info['classes']) * class_counts[class_id])
        else:
            weight = 1.0
        class_weights.append(weight)
    
    print("Poids des classes:")
    for i, (class_name, weight) in enumerate(zip(dataset_info['classes'].values(), class_weights)):
        print(f"  {class_name}: {weight:.3f}")
    
    return class_weights


def create_data_loaders(dataset_dir: str, batch_size: int = 8, num_workers: int = 0):
    """Créer les data loaders pour l'entraînement"""
    
    train_transform = get_transforms('train')
    val_transform = get_transforms('val')
    
    train_dataset = ElementGrossiereDataset(dataset_dir, 'train', train_transform)
    val_dataset = ElementGrossiereDataset(dataset_dir, 'val', val_transform)
    
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
    
    return train_loader, val_loader, train_dataset


def calculate_metrics(predictions, targets, num_classes):
    """Calculer les métriques de segmentation"""
    predictions = torch.argmax(predictions, dim=1)
    
    metrics = {}
    
    # Accuracy globale
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()
    metrics['accuracy'] = accuracy
    
    # IoU par classe
    ious = []
    for class_id in range(num_classes):
        pred_mask = (predictions == class_id)
        target_mask = (targets == class_id)
        
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = torch.tensor(1.0)  # Classe absente dans les deux masques
        
        ious.append(iou.item())
    
    metrics['iou_per_class'] = ious
    metrics['mean_iou'] = np.mean(ious)
    
    return metrics


if __name__ == "__main__":
    """Test du modèle Element Grossiere"""
    
    print("=== Test du modèle Element Grossiere ===")
    
    # Test du modèle
    model = ElementGrossiereUNet(num_classes=4)
    print(f"Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
        print(f"Forme de sortie: {output.shape}")
        print(f"Classes prédites: {torch.argmax(output, dim=1).shape}")
    
    # Test du dataset
    dataset_dir = "data/element_grossiere_dataset"
    if Path(dataset_dir).exists():
        try:
            # Calculer les poids des classes
            class_weights = calculate_class_weights(dataset_dir)
            
            # Créer les data loaders
            train_loader, val_loader, train_dataset = create_data_loaders(dataset_dir)
            
            print(f"Dataset chargé: {len(train_dataset)} images d'entraînement")
            
            # Test d'un batch
            for batch in train_loader:
                print(f"Batch - Image: {batch['image'].shape}, Masque: {batch['mask'].shape}")
                break
                
        except Exception as e:
            print(f"Erreur lors du test du dataset: {e}")
    else:
        print(f"Dataset non trouvé dans {dataset_dir}")
        print("Utiliser d'abord l'outil d'annotation pour créer le dataset.")