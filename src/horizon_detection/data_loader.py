#!/usr/bin/env python3
"""
Chargeur de données pour la détection d'horizons
Traite les annotations LabelMe et extrait les caractéristiques des horizons
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import gabor
from skimage.measure import regionprops
import albumentations as A


class HorizonDataset(Dataset):
    def __init__(self, 
                 images_dir: str,
                 annotations_dir: str,
                 image_size: Tuple[int, int] = (512, 512),
                 augment: bool = False,
                 extract_features: bool = True):
        """
        Dataset pour la détection d'horizons de sol
        
        Args:
            images_dir: Répertoire contenant les images
            annotations_dir: Répertoire contenant les annotations JSON LabelMe
            image_size: Taille de redimensionnement des images
            augment: Appliquer l'augmentation de données
            extract_features: Extraire les caractéristiques de texture et couleur
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.image_size = image_size
        self.extract_features = extract_features
        
        # Obtenir les fichiers correspondants
        self.samples = self._get_sample_pairs()
        
        # Encodeur pour les labels
        self.label_encoder = LabelEncoder()
        self._fit_label_encoder()
        
        # Augmentation de données
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.Resize(image_size[0], image_size[1])
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size[0], image_size[1])
            ])
    
    def _get_sample_pairs(self) -> List[Dict[str, str]]:
        """Obtenir les paires image-annotation correspondantes"""
        samples = []
        
        # Obtenir tous les fichiers JSON d'annotation
        json_files = list(self.annotations_dir.glob("*.json"))
        
        for json_file in json_files:
            # Trouver l'image correspondante
            base_name = json_file.stem.replace('_hz', '')
            
            # Essayer différentes extensions (jpg, JPG)
            image_path = None
            for ext in ['.jpg', '.JPG']:
                candidate_path = self.images_dir / (base_name + ext)
                if candidate_path.exists():
                    image_path = candidate_path
                    break
            
            if image_path:
                samples.append({
                    'image_path': str(image_path),
                    'annotation_path': str(json_file),
                    'image_name': image_path.name
                })
        
        return samples
    
    def _fit_label_encoder(self):
        """Ajuster l'encodeur de labels sur tous les labels disponibles"""
        all_labels = set()
        
        for sample in self.samples:
            with open(sample['annotation_path'], 'r') as f:
                data = json.load(f)
            
            for shape in data.get('shapes', []):
                all_labels.add(shape['label'])
        
        self.label_encoder.fit(list(all_labels))
        print(f"Labels détectés: {self.label_encoder.classes_}")
    
    def _load_annotation(self, annotation_path: str) -> Dict[str, Any]:
        """Charger et traiter les annotations LabelMe"""
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        shapes = data.get('shapes', [])
        
        # Grouper les formes par label
        horizon_polygons = {}
        for shape in shapes:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.float32)
            
            if label not in horizon_polygons:
                horizon_polygons[label] = []
            horizon_polygons[label].append(points)
        
        return horizon_polygons
    
    def _create_line_masks(self, polygons: Dict[str, List], image_shape: Tuple[int, int]) -> np.ndarray:
        """Créer un masque de lignes à partir des polygones - pour la détection d'horizons"""
        # Créer un masque combiné pour toutes les lignes d'horizon
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for label, polygon_list in polygons.items():
            if label == "0":  # Seulement les lignes d'horizon, pas les bordures de fosse
                for polygon in polygon_list:
                    # Dessiner les lignes entre les points consécutifs
                    pts = polygon.astype(np.int32)
                    for i in range(len(pts) - 1):
                        cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 255, thickness=3)
        
        return mask
    
    def _extract_texture_features(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extraire les caractéristiques de texture d'une région"""
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Appliquer le masque
        masked_region = cv2.bitwise_and(gray, gray, mask=mask)
        
        features = []
        
        # 1. Local Binary Pattern (LBP)
        lbp = local_binary_pattern(masked_region, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp[mask > 0], bins=10, range=(0, 10))
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)  # Normalisation
        features.extend(lbp_hist)
        
        # 2. Matrice de cooccurrence (GLCM)
        if np.sum(mask) > 100:  # Vérifier qu'il y a assez de pixels
            glcm = graycomatrix(masked_region, distances=[1, 2], angles=[0, 45, 90, 135], 
                              levels=256, symmetric=True, normed=True)
            
            # Propriétés GLCM
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()

            features.append(contrast.mean())
            features.append(dissimilarity.mean())
            features.append(homogeneity.mean())
            features.append(energy.mean())
        else:
            features.extend([0, 0, 0, 0])
        
        # 3. Filtres de Gabor
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            real, _ = gabor(masked_region, frequency=0.1, theta=np.deg2rad(theta))
            gabor_responses.append(np.mean(real[mask > 0]) if np.sum(mask) > 0 else 0)
        
        features.extend(gabor_responses)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_color_features(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extraire les caractéristiques de couleur d'une région"""
        features = []
        
        # Convertir en différents espaces de couleur
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        color_spaces = [image, hsv, lab]
        
        for color_space in color_spaces:
            for channel in range(3):
                channel_data = color_space[:, :, channel]
                masked_channel = channel_data[mask > 0]
                
                if len(masked_channel) > 0:
                    features.extend([
                        np.mean(masked_channel),
                        np.std(masked_channel),
                        np.median(masked_channel),
                        np.percentile(masked_channel, 25),
                        np.percentile(masked_channel, 75)
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)
    
    def _extract_geometric_features(self, mask: np.ndarray) -> np.ndarray:
        """Extraire les caractéristiques géométriques d'une région"""
        if np.sum(mask) == 0:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Utiliser regionprops pour extraire les caractéristiques
        props = regionprops(mask.astype(int))[0]
        
        features = [
            props.area,
            props.perimeter,
            props.eccentricity,
            props.solidity,
            props.extent,
            props.major_axis_length / (props.minor_axis_length + 1e-8)  # Rapport d'aspect
        ]
        
        return np.array(features, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Charger l'image
        image = cv2.imread(sample['image_path'])
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {sample['image_path']}")
        
        original_shape = image.shape
        
        # Charger les annotations
        polygons = self._load_annotation(sample['annotation_path'])
        
        # Appliquer les transformations
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Calculer les facteurs d'échelle pour ajuster les polygones
        scale_x = self.image_size[1] / original_shape[1]
        scale_y = self.image_size[0] / original_shape[0]
        
        # Ajuster les polygones aux nouvelles dimensions
        scaled_polygons = {}
        for label, polygon_list in polygons.items():
            scaled_polygons[label] = []
            for polygon in polygon_list:
                scaled_polygon = polygon.copy()
                scaled_polygon[:, 0] *= scale_x  # x
                scaled_polygon[:, 1] *= scale_y  # y
                scaled_polygons[label].append(scaled_polygon)
        
        # Créer le masque de lignes d'horizon
        horizon_mask = self._create_line_masks(scaled_polygons, image.shape)
        
        result = {
            'image': image,
            'horizon_mask': horizon_mask,  # Masque binaire des lignes d'horizon
            'polygons': scaled_polygons,
            'image_name': sample['image_name'],
            'original_shape': original_shape
        }
        
        # Pour les modèles de segmentation, convertir le masque en format approprié
        if np.sum(horizon_mask) > 0:
            # Normaliser le masque: 0 = fond, 1 = ligne d'horizon
            result['target_mask'] = (horizon_mask > 0).astype(np.float32)
        else:
            # Masque vide si aucune ligne d'horizon
            result['target_mask'] = np.zeros(image.shape[:2], dtype=np.float32)
        
        return result


def create_horizon_dataloader(images_dir: str,
                            annotations_dir: str,
                            batch_size: int = 4,
                            shuffle: bool = True,
                            augment: bool = False,
                            num_workers: int = 2) -> DataLoader:
    """Créer un DataLoader pour les données d'horizons"""
    
    dataset = HorizonDataset(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x  # Utiliser la fonction de regroupement par défaut
    )
    
    return dataloader


def visualize_sample(dataset: HorizonDataset, idx: int = 0):
    """Visualiser un échantillon du dataset"""
    sample = dataset[idx]
    
    image = sample['image']
    masks = sample['masks']
    
    # Créer la visualisation
    fig, axes = plt.subplots(1, len(masks) + 1, figsize=(15, 5))
    
    # Image originale
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Image originale')
    axes[0].axis('off')
    
    # Masques par label
    for i, (label, mask) in enumerate(masks.items()):
        axes[i + 1].imshow(mask, cmap='gray')
        axes[i + 1].set_title(f'Horizon {label}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les caractéristiques si disponibles
    if 'features' in sample:
        print("\nCaractéristiques extraites par horizon:")
        for label, features in sample['features'].items():
            print(f"Horizon {label}: {len(features)} caractéristiques")
            print(f"  Texture: {features[:18]}")
            print(f"  Couleur: {features[18:63]}")
            print(f"  Géométrie: {features[63:]}")


if __name__ == "__main__":
    # Test du chargeur de données
    images_dir = "/mnt/e/CodeForStudy/Stage/Projet/ACSE_AI_Soil_Image/data/delimitation_horizons"
    annotations_dir = "/mnt/e/CodeForStudy/Stage/Projet/ACSE_AI_Soil_Image/data/horizon"
    
    # Créer le dataset
    dataset = HorizonDataset(images_dir, annotations_dir, augment=True)
    print(f"Dataset créé avec {len(dataset)} échantillons")
    
    # Visualiser un échantillon
    if len(dataset) > 0:
        visualize_sample(dataset, 0)