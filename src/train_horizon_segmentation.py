#!/usr/bin/env python3
"""
Training script for soil horizon segmentation using DeepLabV3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from datetime import datetime
import time
from tqdm import tqdm

from horizon_segmentation_model import (
    HorizonSegmentationModel,
    SoilHorizonDataset, 
    SegmentationLoss,
    create_data_loaders,
    calculate_class_weights
)


class MetricsCalculator:
    """Calculate segmentation metrics"""
    
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch results"""
        # Convert predictions to class indices
        if predictions.dim() == 4:  # (B, C, H, W)
            predictions = torch.argmax(predictions, dim=1)
        
        # Flatten tensors
        predictions = predictions.flatten()
        targets = targets.flatten()
        
        # Remove ignored pixels
        valid_mask = targets != self.ignore_index
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        # Update confusion matrix
        indices = targets * self.num_classes + predictions
        self.confusion_matrix += torch.bincount(
            indices, minlength=self.num_classes**2
        ).view(self.num_classes, self.num_classes)
    
    def compute_metrics(self):
        """Compute final metrics"""
        cm = self.confusion_matrix
        
        # Pixel accuracy
        pixel_acc = torch.diag(cm).sum() / cm.sum()
        
        # Mean accuracy per class
        class_acc = torch.diag(cm) / (cm.sum(dim=1) + 1e-8)
        mean_acc = class_acc.mean()
        
        # IoU per class
        intersection = torch.diag(cm)
        union = cm.sum(dim=0) + cm.sum(dim=1) - intersection
        iou = intersection / (union + 1e-8)
        mean_iou = iou.mean()
        
        return {
            'pixel_accuracy': pixel_acc.item(),
            'mean_accuracy': mean_acc.item(),
            'mean_iou': mean_iou.item(),
            'class_iou': iou.tolist(),
            'class_accuracy': class_acc.tolist()
        }


class HorizonTrainer:
    """Trainer for horizon segmentation model"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, 
                 val_loader: DataLoader, criterion: nn.Module, 
                 optimizer: optim.Optimizer, scheduler=None, 
                 device: str = 'cuda', save_dir: str = 'checkpoints'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        metrics_calc = MetricsCalculator(num_classes=self.model.num_classes)
        
        pbar = tqdm(self.train_loader, desc=f'Époque {epoch+1} - Entraînement')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            
            if isinstance(outputs, dict):
                predictions = outputs['out']
            else:
                predictions = outputs
                
            metrics_calc.update(predictions.detach().cpu(), targets.detach().cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(self.train_loader)
        metrics = metrics_calc.compute_metrics()
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_metrics'].append(metrics)
        
        return avg_loss, metrics
    
    def validate_epoch(self, epoch: int):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        metrics_calc = MetricsCalculator(num_classes=self.model.num_classes)
        
        pbar = tqdm(self.val_loader, desc=f'Époque {epoch+1} - Validation')
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                running_loss += loss.item()
                
                if isinstance(outputs, dict):
                    predictions = outputs['out']
                else:
                    predictions = outputs
                    
                metrics_calc.update(predictions.cpu(), targets.cpu())
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = running_loss / len(self.val_loader)
        metrics = metrics_calc.compute_metrics()
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_metrics'].append(metrics)
        
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_iou': self.best_val_iou
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Nouveau meilleur modèle sauvé: {best_path}")
    
    def train(self, num_epochs: int):
        """Train the model"""
        print(f"Début de l'entraînement pour {num_epochs} époques")
        print(f"Device: {self.device}")
        print(f"Données d'entraînement: {len(self.train_loader.dataset)} échantillons")
        print(f"Données de validation: {len(self.val_loader.dataset)} échantillons")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\n=== Époque {epoch+1}/{num_epochs} ===")
            
            # Training phase
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_metrics['mean_iou']:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_metrics['mean_iou']:.4f}")
            
            # Check if best model
            is_best_loss = val_loss < self.best_val_loss
            is_best_iou = val_metrics['mean_iou'] > self.best_val_iou
            
            if is_best_loss:
                self.best_val_loss = val_loss
            if is_best_iou:
                self.best_val_iou = val_metrics['mean_iou']
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best_iou)
            
            # Save training curves every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_training_curves()
        
        total_time = time.time() - start_time
        print(f"\nEntraînement terminé en {total_time/3600:.2f} heures")
        print(f"Meilleure perte de validation: {self.best_val_loss:.4f}")
        print(f"Meilleur IoU de validation: {self.best_val_iou:.4f}")
        
        # Final plots and save
        self.plot_training_curves()
        self.save_training_history()
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Perte')
        axes[0, 0].set_xlabel('Époque')
        axes[0, 0].set_ylabel('Perte')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # IoU curves
        train_iou = [m['mean_iou'] for m in self.history['train_metrics']]
        val_iou = [m['mean_iou'] for m in self.history['val_metrics']]
        axes[0, 1].plot(epochs, train_iou, 'b-', label='Train')
        axes[0, 1].plot(epochs, val_iou, 'r-', label='Validation')
        axes[0, 1].set_title('IoU Moyen')
        axes[0, 1].set_xlabel('Époque')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Pixel accuracy
        train_acc = [m['pixel_accuracy'] for m in self.history['train_metrics']]
        val_acc = [m['pixel_accuracy'] for m in self.history['val_metrics']]
        axes[1, 0].plot(epochs, train_acc, 'b-', label='Train')
        axes[1, 0].plot(epochs, val_acc, 'r-', label='Validation')
        axes[1, 0].set_title('Précision des Pixels')
        axes[1, 0].set_xlabel('Époque')
        axes[1, 0].set_ylabel('Précision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Mean class accuracy
        train_mean_acc = [m['mean_accuracy'] for m in self.history['train_metrics']]
        val_mean_acc = [m['mean_accuracy'] for m in self.history['val_metrics']]
        axes[1, 1].plot(epochs, train_mean_acc, 'b-', label='Train')
        axes[1, 1].plot(epochs, val_mean_acc, 'r-', label='Validation')
        axes[1, 1].set_title('Précision Moyenne par Classe')
        axes[1, 1].set_xlabel('Époque')
        axes[1, 1].set_ylabel('Précision')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train soil horizon segmentation model')
    parser.add_argument('--data-dir', default='data/processed', help='Path to processed data directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--num-classes', type=int, default=8, help='Number of classes')
    parser.add_argument('--backbone', default='resnet50', choices=['resnet50', 'resnet101'], help='Backbone network')
    parser.add_argument('--image-size', type=int, nargs=2, default=[512, 512], help='Input image size [height width]')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--save-dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--resume', help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Utilisation du device: {device}")
    
    # Create data loaders
    print("Création des chargeurs de données...")
    try:
        train_loader, val_loader, full_dataset = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            image_size=tuple(args.image_size),
            num_workers=args.num_workers
        )
        print(f"Données chargées: {len(train_loader.dataset)} entraînement, {len(val_loader.dataset)} validation")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return
    
    # Create model
    print(f"Création du modèle {args.backbone} avec {args.num_classes} classes...")
    model = HorizonSegmentationModel(
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=True
    )
    
    # Calculate class weights
    print("Calcul des poids de classes...")
    class_weights = calculate_class_weights(full_dataset, args.num_classes)
    class_weights = class_weights.to(device)
    
    # Create loss function
    criterion = SegmentationLoss(weight=class_weights)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create trainer
    trainer = HorizonTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir
    )
    
    # Resume training if checkpoint provided
    start_epoch = 0
    if args.resume:
        print(f"Reprise de l'entraînement depuis {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict'] and scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        trainer.history = checkpoint.get('history', trainer.history)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        trainer.best_val_iou = checkpoint.get('best_val_iou', 0.0)
    
    # Start training
    trainer.train(args.epochs - start_epoch)


if __name__ == "__main__":
    main()