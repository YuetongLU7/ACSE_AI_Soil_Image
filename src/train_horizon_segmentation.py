#!/usr/bin/env python3
"""
Training script for soil horizon depth regression using U-Net
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from datetime import datetime
import time
from tqdm import tqdm

from horizon_segmentation_model import (
    LightweightUNet,
    SoilHorizonDepthDataset, 
    DepthRegressionLoss,
    create_data_loaders,
    calculate_metrics
)


class DepthRegressionTrainer:
    """Trainer for horizon depth regression model"""
    
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 criterion: nn.Module, optimizer: optim.Optimizer, 
                 scheduler=None, device: str = 'cuda', save_dir: str = 'checkpoints'):
        
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
            'train_mae': [],
            'val_mae': [],
            'train_rmse': [],
            'val_rmse': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Époque {epoch+1} - Entraînement')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['depths'].to(self.device)
            valid_mask = batch['valid_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Calculate loss
            loss = self.criterion(predictions, targets, valid_mask)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(predictions, targets, valid_mask)
                
            running_loss += loss.item()
            total_mae += metrics['mae']
            total_rmse += metrics['rmse']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{metrics["mae"]:.2f}cm'
            })
        
        # Calculate epoch metrics
        avg_loss = running_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_rmse = total_rmse / num_batches
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_mae'].append(avg_mae)
        self.history['train_rmse'].append(avg_rmse)
        
        return avg_loss, avg_mae, avg_rmse
    
    def validate_epoch(self, epoch: int):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_batches = 0
        
        pbar = tqdm(self.val_loader, desc=f'Époque {epoch+1} - Validation')
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['depths'].to(self.device)
                valid_mask = batch['valid_mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                loss = self.criterion(predictions, targets, valid_mask)
                
                # Calculate metrics
                metrics = calculate_metrics(predictions, targets, valid_mask)
                
                running_loss += loss.item()
                total_mae += metrics['mae']
                total_rmse += metrics['rmse']
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mae': f'{metrics["mae"]:.2f}cm'
                })
        
        # Calculate epoch metrics
        avg_loss = running_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_rmse = total_rmse / num_batches
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_mae'].append(avg_mae)
        self.history['val_rmse'].append(avg_rmse)
        
        return avg_loss, avg_mae, avg_rmse
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae
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
            train_loss, train_mae, train_rmse = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_mae, val_rmse = self.validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.2f}cm, Train RMSE: {train_rmse:.2f}cm")
            print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.2f}cm, Val RMSE: {val_rmse:.2f}cm")
            
            # Check if best model
            is_best_loss = val_loss < self.best_val_loss
            is_best_mae = val_mae < self.best_val_mae
            
            if is_best_loss:
                self.best_val_loss = val_loss
            if is_best_mae:
                self.best_val_mae = val_mae
            
            # Save checkpoint (use MAE as primary metric)
            self.save_checkpoint(epoch, is_best=is_best_mae)
            
            # Early stopping check
            if val_mae < 5.0:  # If MAE < 5cm, very good performance
                print(f"Excellente performance atteinte (MAE < 5cm), arrêt anticipé!")
                break
            
            # Save training curves every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.plot_training_curves()
        
        total_time = time.time() - start_time
        print(f"\nEntraînement terminé en {total_time/3600:.2f} heures")
        print(f"Meilleure perte de validation: {self.best_val_loss:.4f}")
        print(f"Meilleure MAE de validation: {self.best_val_mae:.2f}cm")
        
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
        
        # MAE curves
        axes[0, 1].plot(epochs, self.history['train_mae'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_mae'], 'r-', label='Validation')
        axes[0, 1].set_title('Erreur Absolue Moyenne (MAE)')
        axes[0, 1].set_xlabel('Époque')
        axes[0, 1].set_ylabel('MAE (cm)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE curves
        axes[1, 0].plot(epochs, self.history['train_rmse'], 'b-', label='Train')
        axes[1, 0].plot(epochs, self.history['val_rmse'], 'r-', label='Validation')
        axes[1, 0].set_title('Erreur Quadratique Moyenne (RMSE)')
        axes[1, 0].set_xlabel('Époque')
        axes[1, 0].set_ylabel('RMSE (cm)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate (if scheduler exists)
        if self.scheduler and hasattr(self.scheduler, 'get_last_lr'):
            try:
                lr_history = [self.scheduler.get_last_lr()[0]] * len(epochs)
                axes[1, 1].plot(epochs, lr_history, 'g-')
                axes[1, 1].set_title('Taux d\'Apprentissage')
                axes[1, 1].set_xlabel('Époque')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].set_yscale('log')
                axes[1, 1].grid(True)
            except:
                axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNon Disponible', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNon Disponible', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train soil horizon depth regression model')
    parser.add_argument('--data-dir', default='data/processed', help='Path to processed data directory')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max-horizons', type=int, default=7, help='Maximum number of horizons')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--save-dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--resume', help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Utilisation du device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create data loaders
    print("Création des chargeurs de données...")
    try:
        train_loader, val_loader, full_dataset = create_data_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers
        )
        print(f"Données chargées: {len(train_loader.dataset)} entraînement, {len(val_loader.dataset)} validation")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return
    
    # Create model
    print(f"Création du modèle U-Net avec {args.max_horizons} horizons maximum...")
    model = LightweightUNet(max_horizons=args.max_horizons)
    print(f"Paramètres du modèle: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function (no class weights needed for regression)
    criterion = DepthRegressionLoss()
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Create trainer
    trainer = DepthRegressionTrainer(
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
        trainer.best_val_mae = checkpoint.get('best_val_mae', float('inf'))
    
    # Start training
    trainer.train(args.epochs - start_epoch)


if __name__ == "__main__":
    main()