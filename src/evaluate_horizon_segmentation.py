#!/usr/bin/env python3
"""
Evaluation and inference script for soil horizon segmentation
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

from horizon_segmentation_model import (
    HorizonSegmentationModel,
    SoilHorizonDataset,
    create_data_loaders,
    get_transforms
)
from train_horizon_segmentation import MetricsCalculator


class HorizonEvaluator:
    """Evaluator for horizon segmentation model"""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Color palette for visualization (distinct colors for each horizon)
        self.colors = [
            [0, 0, 0],        # Background - black
            [128, 0, 0],      # Horizon 1 - dark red
            [0, 128, 0],      # Horizon 2 - dark green  
            [128, 128, 0],    # Horizon 3 - olive
            [0, 0, 128],      # Horizon 4 - navy
            [128, 0, 128],    # Horizon 5 - purple
            [0, 128, 128],    # Horizon 6 - teal
            [192, 192, 192],  # Horizon 7 - silver
        ]
    
    def predict_single_image(self, image_path: str, output_size: tuple = None):
        """
        Predict horizon segmentation for a single image
        
        Args:
            image_path: Path to input image
            output_size: Output size for resizing prediction (H, W)
            
        Returns:
            dict: Prediction results
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        original_size = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        transform = get_transforms('val', (512, 512))
        
        try:
            # Try albumentations format
            transformed = transform(image=image_rgb)
            input_tensor = transformed['image'].unsqueeze(0).to(self.device)
        except:
            # Fall back to PIL format
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)
            input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(input_tensor)
            if isinstance(output, dict):
                predictions = output['out']
            else:
                predictions = output
            
            # Get class probabilities and predictions
            probabilities = F.softmax(predictions, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        
        # Resize to original or specified size
        target_size = output_size if output_size else original_size
        predicted_classes = F.interpolate(
            predicted_classes.unsqueeze(1).float(),
            size=target_size,
            mode='nearest'
        ).squeeze().cpu().numpy().astype(np.uint8)
        
        probabilities = F.interpolate(
            probabilities,
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        return {
            'predictions': predicted_classes,
            'probabilities': probabilities,
            'original_image': image_rgb,
            'input_size': input_tensor.shape[2:],
            'output_size': target_size
        }
    
    def visualize_prediction(self, image: np.ndarray, prediction: np.ndarray, 
                           ground_truth: np.ndarray = None, save_path: str = None):
        """
        Visualize segmentation results
        
        Args:
            image: Original image (H, W, 3)
            prediction: Predicted mask (H, W)
            ground_truth: Ground truth mask (H, W), optional
            save_path: Path to save visualization
        """
        num_plots = 3 if ground_truth is not None else 2
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        
        if num_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Image Originale')
        axes[0].axis('off')
        
        # Predicted segmentation
        pred_colored = self.mask_to_color(prediction)
        axes[1].imshow(pred_colored)
        axes[1].set_title('Prédiction')
        axes[1].axis('off')
        
        # Ground truth (if available)
        if ground_truth is not None:
            gt_colored = self.mask_to_color(ground_truth)
            axes[2].imshow(gt_colored)
            axes[2].set_title('Vérité Terrain')
            axes[2].axis('off')
        
        # Add legend
        unique_classes = np.unique(prediction)
        legend_elements = []
        for class_id in unique_classes:
            if class_id < len(self.colors):
                color = np.array(self.colors[class_id]) / 255.0
                legend_elements.append(plt.Rectangle((0,0),1,1, fc=color, label=f'Horizon {class_id}' if class_id > 0 else 'Arrière-plan'))
        
        if legend_elements:
            axes[-1].legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualisation sauvée: {save_path}")
        
        plt.show()
    
    def mask_to_color(self, mask: np.ndarray):
        """Convert segmentation mask to colored image"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(len(self.colors)):
            colored[mask == class_id] = self.colors[class_id]
        
        return colored
    
    def evaluate_dataset(self, data_loader, save_dir: str = None):
        """
        Evaluate model on dataset
        
        Args:
            data_loader: DataLoader for evaluation
            save_dir: Directory to save results
            
        Returns:
            dict: Evaluation metrics
        """
        metrics_calc = MetricsCalculator(self.model.num_classes)
        all_predictions = []
        all_targets = []
        sample_results = []
        
        print("Évaluation du modèle...")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)
            sample_names = batch['sample_name']
            
            with torch.no_grad():
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    predictions = outputs['out']
                else:
                    predictions = outputs
                
                # Update metrics
                metrics_calc.update(predictions.cpu(), targets.cpu())
                
                # Store results for confusion matrix
                pred_classes = torch.argmax(predictions, dim=1).cpu().numpy()
                target_classes = targets.cpu().numpy()
                
                for i in range(len(sample_names)):
                    all_predictions.extend(pred_classes[i].flatten())
                    all_targets.extend(target_classes[i].flatten())
                    
                    sample_results.append({
                        'sample_name': sample_names[i],
                        'prediction': pred_classes[i],
                        'target': target_classes[i],
                        'image': images[i].cpu()
                    })
        
        # Calculate final metrics
        final_metrics = metrics_calc.compute_metrics()
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions, labels=list(range(self.model.num_classes)))
        
        # Save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            
            # Save metrics
            with open(save_path / 'evaluation_metrics.json', 'w') as f:
                json.dump(final_metrics, f, indent=2)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(cm, save_path / 'confusion_matrix.png')
            
            # Save some sample predictions
            self.save_sample_predictions(sample_results[:10], save_path)
        
        return {
            'metrics': final_metrics,
            'confusion_matrix': cm,
            'sample_results': sample_results
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        class_names = [f'Horizon {i}' if i > 0 else 'Arrière-plan' for i in range(len(cm))]
        df_cm = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
        
        sns.heatmap(df_cm, annot=True, fmt='.2f', cmap='Blues', cbar=True)
        plt.title('Matrice de Confusion Normalisée')
        plt.xlabel('Prédiction')
        plt.ylabel('Vérité Terrain')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matrice de confusion sauvée: {save_path}")
    
    def save_sample_predictions(self, sample_results: list, save_dir: Path):
        """Save sample prediction visualizations"""
        samples_dir = save_dir / 'sample_predictions'
        samples_dir.mkdir(exist_ok=True)
        
        for i, sample in enumerate(sample_results):
            # Convert tensor image back to numpy
            image = sample['image'].permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image * std + mean) * 255
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            save_path = samples_dir / f"{sample['sample_name']}_prediction.png"
            
            self.visualize_prediction(
                image=image,
                prediction=sample['prediction'],
                ground_truth=sample['target'],
                save_path=str(save_path)
            )
            plt.close()  # Close to save memory


def main():
    parser = argparse.ArgumentParser(description='Evaluate soil horizon segmentation model')
    parser.add_argument('--model-path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', default='data/processed', help='Path to processed data directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num-classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--backbone', default='resnet50', choices=['resnet50', 'resnet101'], help='Backbone network')
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 512], help='Input image size [height width]')
    parser.add_argument('--save-dir', default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--single-image', help='Path to single image for prediction')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Utilisation du device: {device}")
    
    # Load model
    print("Chargement du modèle...")
    model = HorizonSegmentationModel(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=False  # Don't need pretrained weights when loading checkpoint
    )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modèle chargé depuis: {args.model_path}")
    
    # Create evaluator
    evaluator = HorizonEvaluator(model, device)
    
    if args.single_image:
        # Single image prediction
        print(f"Prédiction pour image unique: {args.single_image}")
        result = evaluator.predict_single_image(args.single_image)
        
        # Visualize result
        evaluator.visualize_prediction(
            image=result['original_image'],
            prediction=result['predictions'],
            save_path=f"{args.single_image}_prediction.png"
        )
        
        # Print prediction summary
        unique_classes, counts = np.unique(result['predictions'], return_counts=True)
        print("\nRésumé de la prédiction:")
        for class_id, count in zip(unique_classes, counts):
            percentage = (count / result['predictions'].size) * 100
            horizon_name = f"Horizon {class_id}" if class_id > 0 else "Arrière-plan"
            print(f"  {horizon_name}: {percentage:.2f}% ({count} pixels)")
    
    else:
        # Dataset evaluation
        print("Création des chargeurs de données pour l'évaluation...")
        try:
            _, val_loader, _ = create_data_loaders(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                val_split=args.val_split,
                image_size=tuple(args.image_size),
                num_workers=2
            )
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return
        
        # Evaluate model
        results = evaluator.evaluate_dataset(val_loader, args.save_dir)
        
        # Print results
        metrics = results['metrics']
        print("\n=== Résultats d'Évaluation ===")
        print(f"Précision des pixels: {metrics['pixel_accuracy']:.4f}")
        print(f"Précision moyenne: {metrics['mean_accuracy']:.4f}")
        print(f"IoU moyen: {metrics['mean_iou']:.4f}")
        
        print("\nIoU par classe:")
        for i, iou in enumerate(metrics['class_iou']):
            class_name = f"Horizon {i}" if i > 0 else "Arrière-plan"
            print(f"  {class_name}: {iou:.4f}")
        
        print(f"\nRésultats sauvés dans: {args.save_dir}")


if __name__ == "__main__":
    main()