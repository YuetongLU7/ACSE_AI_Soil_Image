#!/usr/bin/env python3
"""
Prediction script for Element Grossiere segmentation
Script de prédiction pour la segmentation des éléments grossiers
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
import torchvision.transforms as transforms

from src.element_grossiere_model import ElementGrossiereUNet, load_mask_from_json_static


def load_model(checkpoint_path: str, num_classes: int = 3, device: str = 'cuda'):
    """Charger le modèle depuis un checkpoint"""
    model = ElementGrossiereUNet(num_classes=num_classes)
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modèle chargé depuis: {checkpoint_path}")
    else:
        print(f"Checkpoint non trouvé: {checkpoint_path}")
        print("Utilisation du modèle non entraîné")
    
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path: str, size: tuple = (512, 512)):
    """Préprocesser l'image pour l'inférence"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    
    # Transformer l'image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor, image, original_shape


def postprocess_prediction(prediction: torch.Tensor, original_shape: tuple):
    """Post-traiter la prédiction"""
    # Prendre la classe avec la probabilité maximale
    pred_mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    
    # Redimensionner à la taille originale
    pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                          (original_shape[1], original_shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    return pred_mask


def create_overlay_mask(original_image: np.ndarray, mask: np.ndarray):
    """创建叠加掩码，只标注石头和根系"""
    overlay = original_image.copy()
    
    # 石头用红色标注
    stone_mask = (mask == 1)
    overlay[stone_mask] = [255, 0, 0]  # 红色
    
    # 根系用绿色标注  
    root_mask = (mask == 2)
    overlay[root_mask] = [0, 255, 0]  # 绿色
    
    return overlay


def visualize_results(original_image: np.ndarray, 
                     ground_truth: np.ndarray, 
                     prediction: np.ndarray,
                     image_name: str,
                     save_dir: Path):
    """Visualiser les résultats - seulement les overlays"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Image originale
    axes[0].imshow(original_image)
    axes[0].set_title('Image Originale', fontsize=14, pad=20)
    axes[0].axis('off')
    
    # 真实标注叠加在原图上
    gt_overlay = create_overlay_mask(original_image, ground_truth)
    axes[1].imshow(gt_overlay)
    axes[1].set_title('Annotation Manuelle', fontsize=14, pad=20)
    axes[1].axis('off')
    
    # 模型预测叠加在原图上
    pred_overlay = create_overlay_mask(original_image, prediction)
    axes[2].imshow(pred_overlay)
    axes[2].set_title('Prediction Modele\n(Rouge=Pierre, Vert=Racine)', fontsize=14, pad=20)
    axes[2].axis('off')
    
    # 添加类别标签说明（去掉背景）
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='1: Pierre'),
        plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='2: Racine')
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=12)
    
    plt.suptitle(f'Resultats pour: {image_name}', fontsize=16, y=0.95)
    plt.tight_layout()


def predict_all_images():
    """Prédire toutes les images du dataset"""
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device utilisé: {device}")
    
    # Créer le dossier de résultats
    results_dir = Path("results/predictions")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Charger la configuration du dataset
    with open("data/dataset_info.json", 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    checkpoint_path = "checkpoints/element_grossiere/checkpoint_epoch_19.pth"
    
    # Charger le modèle
    model = load_model(checkpoint_path, num_classes=3, device=device)
    
    print(f"\n=== Prédiction pour {len(dataset_info['images'])} images ===\n")
    
    for i, image_info in enumerate(dataset_info['images']):
        print(f"Traitement de l'image {i+1}/{len(dataset_info['images'])}")
        
        # Chemins
        image_path = f"data/{image_info['image_path']}"
        annotation_path = f"data/{image_info['annotation_path']}"
        image_name = Path(image_info['image_path']).stem
        
        try:
            # Préprocesser l'image
            image_tensor, original_image, original_shape = preprocess_image(image_path)
            image_tensor = image_tensor.to(device)
            
            # Charger la vérité terrain
            ground_truth = load_mask_from_json_static(Path(annotation_path), original_shape)
            
            # Faire la prédiction
            with torch.no_grad():
                prediction_tensor = model(image_tensor)
                prediction = postprocess_prediction(prediction_tensor, original_shape)
            
            # Calculer les métriques
            accuracy = np.mean(prediction == ground_truth)
            print(f"  Précision pixel par pixel: {accuracy:.3f}")
            
            # Afficher les statistiques par classe
            class_names = ['Arriere-plan', 'Pierre', 'Racine']
            for class_id in range(3):
                gt_pixels = np.sum(ground_truth == class_id)
                pred_pixels = np.sum(prediction == class_id)
                correct_pixels = np.sum((ground_truth == class_id) & (prediction == class_id))
                
                if gt_pixels > 0:
                    recall = correct_pixels / gt_pixels
                    precision = correct_pixels / pred_pixels if pred_pixels > 0 else 0
                    print(f"  {class_names[class_id]}: Recall={recall:.3f}, Precision={precision:.3f}")
            
            # Visualiser et sauvegarder les résultats
            save_path = results_dir / f"prediction_{image_name}.png"
            visualize_results(original_image, ground_truth, prediction, image_name, results_dir)
            
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight', pad_inches=0.2)
            plt.close()  # Fermer la figure pour libérer la mémoire
            
            print(f"  Résultats sauvegardés: {save_path}")
            print()
            
        except Exception as e:
            print(f"  Erreur pour {image_name}: {e}")
            print()
            continue
    
    print(f"Toutes les prédictions terminées! Résultats dans: {results_dir}")


def main():
    predict_all_images()


if __name__ == "__main__":
    main()