#!/usr/bin/env python3
"""
Modèle de détection d'horizons de sol basé sur les caractéristiques de texture et couleur
Utilise une approche hybride: CNN pour les caractéristiques spatiales + caractéristiques extraites manuellement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Any
import cv2
from pathlib import Path
import torchvision.models as models
import segmentation_models_pytorch as smp
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib


class TextureColorCNN(nn.Module):
    """CNN pour extraire les caractéristiques de texture et couleur"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(TextureColorCNN, self).__init__()
        
        # Utiliser ResNet comme backbone
        self.backbone = models.resnet34(pretrained=pretrained)
        
        # Remplacer la première couche pour accepter différents canaux
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extraire les caractéristiques avant la couche finale
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Couches pour l'analyse multi-échelle
        self.texture_branch = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.color_branch = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # Couche de fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # Classificateur final
        self.classifier = nn.Sequential(
            nn.Conv2d(64, num_classes, kernel_size=1),
        )
        
        # Pooling adaptatif global pour les caractéristiques
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        # Extraire les caractéristiques de base
        features = self.feature_extractor(x)
        
        # Branches spécialisées
        texture_feat = self.texture_branch(features)
        color_feat = self.color_branch(features)
        
        # Fusion des caractéristiques
        combined = torch.cat([texture_feat, color_feat], dim=1)
        fused = self.fusion(combined)
        
        # Classification par pixel
        output = self.classifier(fused)
        
        # Redimensionner à la taille d'entrée
        output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return output
    
    def extract_features(self, x):
        """Extraire les caractéristiques pour analyse"""
        with torch.no_grad():
            features = self.feature_extractor(x)
            texture_feat = self.texture_branch(features)
            color_feat = self.color_branch(features)
            
            # Pooling global pour obtenir un vecteur de caractéristiques
            texture_vec = self.global_pool(texture_feat).flatten(1)
            color_vec = self.global_pool(color_feat).flatten(1)
            
            return torch.cat([texture_vec, color_vec], dim=1)


class HorizonUNet(nn.Module):
    """U-Net spécialisé pour la segmentation d'horizons"""
    
    def __init__(self, num_classes: int = 2, encoder_name: str = 'resnet34'):
        super(HorizonUNet, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
            activation=None  # Pas d'activation finale
        )
        
        # Couche d'attention pour les horizons
        self.attention = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output = self.model(x)
        
        # Appliquer l'attention
        attention_weights = self.attention(output)
        output = output * attention_weights
        
        return output


class HybridHorizonDetector(nn.Module):
    """Détecteur d'horizons hybride combinant CNN et caractéristiques extraites"""
    
    def __init__(self, 
                 num_classes: int = 2,
                 feature_dim: int = 69,  # Dimension des caractéristiques extraites
                 use_cnn: bool = True,
                 use_features: bool = True):
        super(HybridHorizonDetector, self).__init__()
        
        self.use_cnn = use_cnn
        self.use_features = use_features
        
        if use_cnn:
            self.cnn_branch = HorizonUNet(num_classes)
        
        if use_features:
            # Branche pour les caractéristiques extraites
            self.feature_branch = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        # Couche de fusion si les deux branches sont utilisées
        if use_cnn and use_features:
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, image, features=None):
        outputs = []
        
        if self.use_cnn:
            cnn_output = self.cnn_branch(image)
            outputs.append(cnn_output)
        
        if self.use_features and features is not None:
            feature_output = self.feature_branch(features)
            outputs.append(feature_output)
        
        if len(outputs) == 1:
            return outputs[0]
        elif len(outputs) == 2:
            # Fusion pondérée
            alpha = torch.sigmoid(self.fusion_weight)
            # Redimensionner les sorties des caractéristiques pour correspondre à l'image
            feature_output_resized = outputs[1].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, outputs[0].size(2), outputs[0].size(3))
            return alpha * outputs[0] + (1 - alpha) * feature_output_resized
        
        return None


class HorizonClassifier:
    """Classificateur d'horizons utilisant des méthodes traditionnelles ML"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                gamma='scale',
                class_weight='balanced',
                probability=True
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    def train(self, features: List[np.ndarray], labels: List[str]):
        """Entraîner le classificateur"""
        X = np.array(features)
        y = np.array(labels)
        
        print(f"Entraînement avec {len(X)} échantillons")
        print(f"Forme des caractéristiques: {X.shape}")
        print(f"Labels uniques: {np.unique(y)}")
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Évaluation sur les données d'entraînement
        y_pred = self.model.predict(X)
        print("\nRapport de classification (entraînement):")
        print(classification_report(y, y_pred))
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Prédire les labels"""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant la prédiction")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        return self.model.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Prédire les probabilités"""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant la prédiction")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        return self.model.predict_proba(features)
    
    def save(self, path: str):
        """Sauvegarder le modèle"""
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant la sauvegarde")
        
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }, path)
        print(f"Modèle sauvegardé dans: {path}")
    
    def load(self, path: str):
        """Charger le modèle"""
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.is_trained = data['is_trained']
        print(f"Modèle chargé depuis: {path}")


class HorizonLoss(nn.Module):
    """Fonction de perte spécialisée pour la détection d'horizons"""
    
    def __init__(self, 
                 class_weights: torch.Tensor = None,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0,
                 dice_weight: float = 0.5):
        super(HorizonLoss, self).__init__()
        
        self.class_weights = class_weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        
    def focal_loss(self, inputs, targets):
        """Focal Loss pour gérer le déséquilibre de classes"""
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, inputs, targets, smooth=1e-6):
        """Dice Loss pour la segmentation"""
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=inputs.size(1))
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return (1 - self.dice_weight) * focal + self.dice_weight * dice


def create_horizon_model(model_type: str = 'hybrid', 
                        num_classes: int = 2,
                        **kwargs) -> nn.Module:
    """Factory pour créer différents types de modèles"""
    
    if model_type == 'cnn':
        return TextureColorCNN(num_classes, **kwargs)
    elif model_type == 'unet':
        return HorizonUNet(num_classes, **kwargs)
    elif model_type == 'hybrid':
        return HybridHorizonDetector(num_classes, **kwargs)
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")


if __name__ == "__main__":
    # Test des modèles
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    # Test du modèle CNN
    model = create_horizon_model('cnn', num_classes=3)
    model.to(device)
    
    # Test avec une entrée factice
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 512, 512).to(device)
    
    print("Test du modèle CNN...")
    with torch.no_grad():
        output = model(test_input)
        print(f"Forme de sortie: {output.shape}")
    
    # Test du classificateur traditionnel
    print("\nTest du classificateur Random Forest...")
    classifier = HorizonClassifier('random_forest')
    
    # Données factices
    fake_features = [np.random.randn(69) for _ in range(100)]
    fake_labels = [str(i % 3) for i in range(100)]
    
    classifier.train(fake_features, fake_labels)
    
    # Test de prédiction
    test_features = np.random.randn(5, 69)
    predictions = classifier.predict(test_features)
    print(f"Prédictions: {predictions}")