#!/usr/bin/env python3
"""
Script de prédiction pour la détection d'horizons
Utilise les modèles entraînés pour prédire les horizons sur de nouvelles images
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from skimage.measure import regionprops, label
import albumentations as A

# Imports locaux
from horizon_model import create_horizon_model, HorizonClassifier
from data_loader import HorizonDataset


class HorizonPredictor:
    """Prédicteur d'horizons de sol"""
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = 'hybrid',
                 device: str = 'auto',
                 image_size: Tuple[int, int] = (512, 512)):
        
        self.model_path = model_path
        self.model_type = model_type
        self.image_size = image_size
        
        # Configuration du device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Utilisation du device: {self.device}")
        
        # Charger le modèle
        self.model = None
        self.load_model()
        
        # Transform pour les images
        self.transform = A.Compose([
            A.Resize(image_size[0], image_size[1])
        ])
    
    def load_model(self):
        """Charger le modèle pré-entraîné"""
        print(f"Chargement du modèle depuis: {self.model_path}")
        
        if self.model_type in ['cnn', 'unet', 'hybrid']:
            # Modèles deep learning
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Créer le modèle
            num_classes = checkpoint.get('num_classes', 3)
            self.model = create_horizon_model(self.model_type, num_classes)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Modèle {self.model_type} chargé avec {num_classes} classes")
            
        elif self.model_type in ['random_forest', 'svm']:
            # Modèles traditionnels
            self.model = HorizonClassifier(self.model_type)
            self.model.load(self.model_path)
            
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")
    
    def _extract_texture_features(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Extraire les caractéristiques de texture"""
        # Convertir en niveaux de gris si nécessaire
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=mask)
        
        features = []
        
        # 1. Local Binary Pattern (LBP)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        if mask is not None:
            lbp_hist, _ = np.histogram(lbp[mask > 0], bins=10, range=(0, 10))
        else:
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)
        features.extend(lbp_hist)
        
        # 2. GLCM
        valid_pixels = np.sum(mask) if mask is not None else gray.size
        if valid_pixels > 100:
            glcm = graycomatrix(gray, distances=[1, 2], angles=[0, 45, 90, 135], 
                              levels=256, symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            
            features.extend([contrast.mean(), dissimilarity.mean(), 
                           homogeneity.mean(), energy.mean()])
        else:
            features.extend([0, 0, 0, 0])
        
        # 3. Filtres de Gabor
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            real, _ = gabor(gray, frequency=0.1, theta=np.deg2rad(theta))
            if mask is not None:
                response = np.mean(real[mask > 0]) if np.sum(mask) > 0 else 0
            else:
                response = np.mean(real)
            gabor_responses.append(response)
        
        features.extend(gabor_responses)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_color_features(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """Extraire les caractéristiques de couleur"""
        features = []
        
        # Convertir en différents espaces de couleur
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        color_spaces = [image, hsv, lab]
        
        for color_space in color_spaces:
            for channel in range(3):
                channel_data = color_space[:, :, channel]
                
                if mask is not None:
                    masked_channel = channel_data[mask > 0]
                else:
                    masked_channel = channel_data.flatten()
                
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
        """Extraire les caractéristiques géométriques"""
        if np.sum(mask) == 0:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        props = regionprops(mask.astype(int))[0]
        
        features = [
            props.area,
            props.perimeter,
            props.eccentricity,
            props.solidity,
            props.extent,
            props.major_axis_length / (props.minor_axis_length + 1e-8)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def extract_features_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extraire toutes les caractéristiques d'une image"""
        texture_feat = self._extract_texture_features(image)
        color_feat = self._extract_color_features(image)
        
        # Pour les caractéristiques géométriques, utiliser un masque de l'image entière
        full_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        geom_feat = self._extract_geometric_features(full_mask)
        
        combined_features = np.concatenate([texture_feat, color_feat, geom_feat])
        return combined_features
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """Prédire les horizons sur une image"""
        
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        original_shape = image.shape
        print(f"Image chargée: {original_shape}")
        
        # Redimensionner l'image
        transformed = self.transform(image=image)
        resized_image = transformed['image']
        
        results = {}
        
        if self.model_type in ['cnn', 'unet', 'hybrid']:
            results = self._predict_deep_learning(resized_image, original_shape)
        elif self.model_type in ['random_forest', 'svm']:
            results = self._predict_traditional(resized_image)
        
        results['image_path'] = image_path
        results['original_shape'] = original_shape
        results['processed_shape'] = resized_image.shape
        
        return results
    
    def _predict_deep_learning(self, image: np.ndarray, original_shape: Tuple) -> Dict[str, Any]:
        """Prédiction avec modèles deep learning"""
        
        # Convertir en tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Ajouter dimension batch
        
        with torch.no_grad():
            if self.model_type == 'hybrid':
                # Extraire les caractéristiques pour le modèle hybride
                features = self.extract_features_from_image(image)
                features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
                output = self.model(image_tensor, features_tensor)
            else:
                output = self.model(image_tensor)
            
            # Appliquer softmax pour obtenir les probabilités
            probabilities = F.softmax(output, dim=1)
            
            # Prédiction de classe avec seuil adaptatif pour les lignes d'horizon
            # Utiliser un seuil plus bas (0.2) au lieu d'argmax pour détecter les lignes fines
            class1_probs = probabilities[:, 1]  # Probabilités de la classe horizon
            pred_mask = (class1_probs > 0.2).long()
            
            # Convertir en numpy
            pred_mask_np = pred_mask.cpu().numpy()[0]
            probs_np = probabilities.cpu().numpy()[0]
        
        # Redimensionner à la taille originale
        pred_mask_resized = cv2.resize(
            pred_mask_np.astype(np.uint8), 
            (original_shape[1], original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        return {
            'prediction_mask': pred_mask_resized,
            'probabilities': probs_np,
            'model_type': self.model_type
        }
    
    def _predict_traditional(self, image: np.ndarray) -> Dict[str, Any]:
        """Prédiction avec modèles traditionnels"""
        
        # Extraire les caractéristiques de l'image entière
        features = self.extract_features_from_image(image)
        
        # Prédiction
        pred_class = self.model.predict(features.reshape(1, -1))[0]
        pred_probs = self.model.predict_proba(features.reshape(1, -1))[0]
        
        return {
            'predicted_class': pred_class,
            'class_probabilities': pred_probs,
            'features': features,
            'model_type': self.model_type
        }
    
    def _analyze_predicted_regions(self, pred_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Analyser les régions prédites"""
        regions = []
        
        # Pour chaque classe prédite
        unique_classes = np.unique(pred_mask)
        print(f"Classes uniques détectées: {unique_classes}")
        
        for class_id in unique_classes:
            # Ne pas ignorer la classe 0 si c'est la seule classe prédite
            # if class_id == 0:  # Ignorer le fond
            #     continue
            
            # Créer un masque binaire pour cette classe
            class_mask = (pred_mask == class_id).astype(np.uint8) * 255
            
            # Trouver les contours
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) < 100:  # Filtrer les petites régions
                    continue
                
                # Obtenir les propriétés de la région
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Simplifier le contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                region = {
                    'class_id': int(class_id),
                    'region_id': i,
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'area': float(area),
                    'perimeter': float(perimeter),
                    'contour': approx.reshape(-1, 2).tolist(),
                    'center': [int(x + w//2), int(y + h//2)]
                }
                
                regions.append(region)
        
        return regions
    
    def visualize_prediction(self, image_path: str, results: Dict[str, Any], save_path: str = None):
        """Visualiser les résultats de prédiction"""
        
        # Charger l'image originale
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.model_type in ['cnn', 'unet', 'hybrid']:
            self._visualize_segmentation_results(image_rgb, results, save_path)
        else:
            self._visualize_classification_results(image_rgb, results, save_path)
    
    def _visualize_segmentation_results(self, image: np.ndarray, results: Dict[str, Any], save_path: str = None):
        """Visualiser les résultats de détection de lignes d'horizon"""
        
        pred_mask = results['prediction_mask']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Image originale
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Image originale')
        axes[0].axis('off')
        
        # Masque de lignes d'horizon prédites
        axes[1].imshow(pred_mask, cmap='gray')
        axes[1].set_title('Lignes d\'horizon détectées')
        axes[1].axis('off')
        
        # Superposition des lignes sur l'image originale
        axes[2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Superposer les lignes d'horizon en rouge
        horizon_pixels = np.where(pred_mask > 0)
        if len(horizon_pixels[0]) > 0:
            # Créer une image overlay rouge pour les horizons
            overlay = image.copy()
            overlay[horizon_pixels] = [0, 0, 255]  # Rouge en BGR
            
            # Mélanger avec l'image originale
            alpha = 0.7
            blended = cv2.addWeighted(image, alpha, overlay, 1-alpha, 0)
            axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        
        axes[2].set_title('Horizons superposés')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvegardée: {save_path}")
        
        plt.show()
        
        # Statistiques
        horizon_pixel_count = np.sum(pred_mask > 0)
        total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
        print(f"\nStatistiques de détection:")
        print(f"Pixels d'horizon détectés: {horizon_pixel_count}")
        print(f"Pourcentage de l'image: {100 * horizon_pixel_count / total_pixels:.2f}%")
        
        return fig
    
    def _visualize_classification_results(self, image: np.ndarray, results: Dict[str, Any], save_path: str = None):
        """Visualiser les résultats de classification"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Image originale
        ax1.imshow(image)
        ax1.set_title(f'Classe prédite: {results["predicted_class"]}')
        ax1.axis('off')
        
        # Probabilités des classes
        probs = results['class_probabilities']
        classes = range(len(probs))
        
        ax2.bar(classes, probs)
        ax2.set_title('Probabilités par classe')
        ax2.set_xlabel('Classe')
        ax2.set_ylabel('Probabilité')
        ax2.set_ylim(0, 1)
        
        # Ajouter les valeurs sur les barres
        for i, prob in enumerate(probs):
            ax2.text(i, prob + 0.01, f'{prob:.3f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvegardée: {save_path}")
        
        plt.show()
    
    def predict_batch(self, image_dir: str, output_dir: str = None) -> List[Dict[str, Any]]:
        """Prédire sur un lot d'images"""
        
        image_dir = Path(image_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        # Obtenir toutes les images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        results = []
        
        print(f"Traitement de {len(image_files)} images...")
        
        for image_file in image_files:
            print(f"Traitement: {image_file.name}")
            
            try:
                # Prédiction
                result = self.predict_image(str(image_file))
                result['image_name'] = image_file.name
                results.append(result)
                
                # Sauvegarder la visualisation si un répertoire de sortie est spécifié
                if output_dir:
                    save_path = output_dir / f"{image_file.stem}_prediction.png"
                    self.visualize_prediction(str(image_file), result, str(save_path))
                
            except Exception as e:
                print(f"Erreur lors du traitement de {image_file.name}: {e}")
                continue
        
        # Sauvegarder les résultats en JSON
        if output_dir:
            results_file = output_dir / "prediction_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                # Convertir les arrays numpy en listes pour la sérialisation JSON
                serializable_results = []
                for result in results:
                    serializable_result = result.copy()
                    if 'prediction_mask' in serializable_result:
                        serializable_result['prediction_mask'] = serializable_result['prediction_mask'].tolist()
                    if 'probabilities' in serializable_result:
                        serializable_result['probabilities'] = serializable_result['probabilities'].tolist()
                    if 'features' in serializable_result:
                        serializable_result['features'] = serializable_result['features'].tolist()
                    serializable_results.append(serializable_result)
                
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            print(f"Résultats sauvegardés: {results_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Prédire les horizons sur des images")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Chemin vers le modèle entraîné")
    parser.add_argument("--model_type", type=str, default="hybrid",
                       choices=['cnn', 'unet', 'hybrid', 'random_forest', 'svm'],
                       help="Type de modèle")
    parser.add_argument("--image_path", type=str,
                       help="Chemin vers une image (pour prédiction unique)")
    parser.add_argument("--image_dir", type=str,
                       help="Répertoire d'images (pour prédiction en lot)")
    parser.add_argument("--output_dir", type=str,
                       help="Répertoire de sortie pour les résultats")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device à utiliser (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    if not args.image_path and not args.image_dir:
        print("Erreur: Spécifiez soit --image_path soit --image_dir")
        return
    
    # Créer le prédicteur
    predictor = HorizonPredictor(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device
    )
    
    if args.image_path:
        # Prédiction sur une seule image
        results = predictor.predict_image(args.image_path)
        
        # Visualiser les résultats
        save_path = None
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            image_name = Path(args.image_path).stem
            save_path = output_dir / f"{image_name}_prediction.png"
        
        predictor.visualize_prediction(args.image_path, results, save_path)
    
    elif args.image_dir:
        # Prédiction en lot
        results = predictor.predict_batch(args.image_dir, args.output_dir)
        
        print(f"\nPrédiction terminée sur {len(results)} images")


if __name__ == "__main__":
    main()