#!/usr/bin/env python3
"""
Training script for Element Grossiere segmentation
Script d'entraînement pour la segmentation des éléments grossiers
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.element_grossiere_model import (
    ElementGrossiereUNet, 
    create_data_loaders, 
    WeightedCrossEntropyLoss,
    calculate_class_weights,
    calculate_metrics
)


class ElementGrossiereTrainer:
    """Entraîneur pour le modèle Element Grossiere"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Utilisation du device: {self.device}")
        
        # Créer les répertoires
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(config.get('tensorboard_dir', 'runs/element_grossiere'))
        
        # Initialiser le modèle
        self.model = ElementGrossiereUNet(num_classes=config['num_classes'])
        self.model.to(self.device)
        
        # Créer les data loaders
        self.train_loader, self.val_loader, self.train_dataset = create_data_loaders(
            config['dataset_dir'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 0)
        )
        
        # Calculer les poids des classes
        if config.get('use_class_weights', True):
            class_weights = calculate_class_weights(config['dataset_dir'])
            self.criterion = WeightedCrossEntropyLoss(class_weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()
        
        self.criterion.to(self.device)
        
        # Optimiseur
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 10)
        )
        
        # Historique d'entraînement
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_miou': [],
            'val_miou': []
        }
        
        self.best_val_miou = 0.0
        self.start_epoch = 0
    
    def train_epoch(self, epoch):
        """Entraîner une époque"""
        self.model.train()
        train_loss = 0.0
        train_metrics = {'accuracy': 0.0, 'mean_iou': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Métriques
            with torch.no_grad():
                batch_metrics = calculate_metrics(outputs, targets, self.config['num_classes'])
                train_metrics['accuracy'] += batch_metrics['accuracy']
                train_metrics['mean_iou'] += batch_metrics['mean_iou']
            
            train_loss += loss.item()
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{batch_metrics["accuracy"]:.3f}',
                'mIoU': f'{batch_metrics["mean_iou"]:.3f}'
            })
        
        # Moyennes
        train_loss /= len(self.train_loader)
        train_metrics['accuracy'] /= len(self.train_loader)
        train_metrics['mean_iou'] /= len(self.train_loader)
        
        return train_loss, train_metrics
    
    def validate_epoch(self, epoch):
        """Valider une époque"""
        self.model.eval()
        val_loss = 0.0
        val_metrics = {'accuracy': 0.0, 'mean_iou': 0.0}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                batch_metrics = calculate_metrics(outputs, targets, self.config['num_classes'])
                
                val_loss += loss.item()
                val_metrics['accuracy'] += batch_metrics['accuracy']
                val_metrics['mean_iou'] += batch_metrics['mean_iou']
        
        # Moyennes
        val_loss /= len(self.val_loader)
        val_metrics['accuracy'] /= len(self.val_loader)
        val_metrics['mean_iou'] /= len(self.val_loader)
        
        return val_loss, val_metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Sauvegarder un checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_miou': self.best_val_miou,
            'train_history': self.train_history,
            'config': self.config
        }
        
        # Checkpoint régulier
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Meilleur modèle
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Nouveau meilleur modèle sauvegardé (mIoU: {self.best_val_miou:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """Charger un checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_miou = checkpoint['best_val_miou']
        self.train_history = checkpoint['train_history']
        self.start_epoch = checkpoint['epoch'] + 1
        
        print(f"Checkpoint chargé depuis l'époque {checkpoint['epoch']}")
    
    def train(self):
        """Boucle d'entraînement principale"""
        print("=== Début de l'entraînement ===")
        print(f"Dataset: {len(self.train_dataset)} images d'entraînement")
        print(f"Validation: {len(self.val_loader.dataset)} images")
        print(f"Device: {self.device}")
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # Entraînement
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(epoch)
            
            # Scheduler
            self.scheduler.step(val_loss)
            
            # Enregistrer l'historique
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_accuracy'].append(train_metrics['accuracy'])
            self.train_history['val_accuracy'].append(val_metrics['accuracy'])
            self.train_history['train_miou'].append(train_metrics['mean_iou'])
            self.train_history['val_miou'].append(val_metrics['mean_iou'])
            
            # TensorBoard
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('mIoU/Train', train_metrics['mean_iou'], epoch)
            self.writer.add_scalar('mIoU/Validation', val_metrics['mean_iou'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Afficher les résultats
            print(f"\nÉpoque {epoch}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.3f}, mIoU: {train_metrics['mean_iou']:.3f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.3f}, mIoU: {val_metrics['mean_iou']:.3f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Sauvegarder le meilleur modèle
            is_best = val_metrics['mean_iou'] > self.best_val_miou
            if is_best:
                self.best_val_miou = val_metrics['mean_iou']
            
            # Sauvegarder les checkpoints
            if (epoch + 1) % self.config.get('save_frequency', 5) == 0:
                self.save_checkpoint(epoch, is_best)
        
        # Sauvegarder le modèle final
        self.save_checkpoint(self.config['num_epochs'] - 1, is_best)
        
        # Créer les graphiques d'entraînement
        self.plot_training_curves()
        
        print("=== Entraînement terminé ===")
        print(f"Meilleur mIoU de validation: {self.best_val_miou:.4f}")
    
    def plot_training_curves(self):
        """Créer les graphiques des courbes d'entraînement"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.train_history['train_loss']))
        
        # Loss
        axes[0, 0].plot(epochs, self.train_history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.train_history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.train_history['train_accuracy'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.train_history['val_accuracy'], 'r-', label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # mIoU
        axes[1, 0].plot(epochs, self.train_history['train_miou'], 'b-', label='Train')
        axes[1, 0].plot(epochs, self.train_history['val_miou'], 'r-', label='Validation')
        axes[1, 0].set_title('Mean IoU')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('mIoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        lr_history = [step_data for step_data in self.train_history.get('learning_rates', [])]
        if lr_history:
            axes[1, 1].plot(epochs[:len(lr_history)], lr_history, 'g-')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'LR history not available', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sauvegarder l'historique
        with open(self.checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(self.train_history, f, indent=2)


def create_config():
    """Créer la configuration par défaut"""
    return {
        'dataset_dir': 'data',
        'checkpoint_dir': 'checkpoints/element_grossiere',
        'tensorboard_dir': 'runs/element_grossiere',
        'num_classes': 3,
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler_patience': 10,
        'save_frequency': 5,
        'num_workers': 0,
        'use_class_weights': True
    }


def main():
    parser = argparse.ArgumentParser(description='Entraînement Element Grossiere')
    parser.add_argument('--dataset', required=True, help='Répertoire du dataset')
    parser.add_argument('--config', help='Fichier de configuration JSON')
    parser.add_argument('--resume', help='Checkpoint à reprendre')
    parser.add_argument('--epochs', type=int, default=100, help='Nombre d\'époques')
    parser.add_argument('--batch-size', type=int, default=8, help='Taille du batch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    
    args = parser.parse_args()
    
    # Charger la configuration
    config = create_config()
    config['dataset_dir'] = args.dataset
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['learning_rate'] = args.lr
    
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Vérifier que le dataset existe
    if not Path(config['dataset_dir']).exists():
        print(f"Erreur: Dataset non trouvé dans {config['dataset_dir']}")
        print("Utilisez d'abord l'outil d'annotation pour créer le dataset:")
        print("python tools/element_grossiere_annotator.py --mode prepare --input data/processed --output annotation_dir")
        return
    
    # Créer l'entraîneur
    trainer = ElementGrossiereTrainer(config)
    
    # Reprendre l'entraînement si demandé
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Lancer l'entraînement
    trainer.train()


if __name__ == "__main__":
    main()