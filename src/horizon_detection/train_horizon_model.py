#!/usr/bin/env python3
"""
Script d'entraînement pour les modèles de détection d'horizons
Supporte l'entraînement des modèles CNN, U-Net, hybrides et classificateurs traditionnels
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import cv2
from pathlib import Path
import argparse
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Imports locaux
from data_loader import HorizonDataset, create_horizon_dataloader
from horizon_model import (
    create_horizon_model, 
    HorizonClassifier, 
    HorizonLoss,
    TextureColorCNN,
    HorizonUNet,
    HybridHorizonDetector
)


class HorizonTrainer:
    """Entraîneur pour les modèles de détection d'horizons"""
    
    def __init__(self, 
                 model_type: str = 'hybrid',
                 num_classes: int = 3,
                 device: str = 'auto',
                 learning_rate: float = 1e-3,
                 batch_size: int = 4,
                 num_epochs: int = 50):
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Configuration du device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Utilisation du device: {self.device}")
        
        # Initialiser le modèle
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Historique d'entraînement
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def setup_model(self):
        """Configurer le modèle et les composants d'entraînement"""
        
        if self.model_type in ['cnn', 'unet', 'hybrid']:
            # Modèles deep learning
            self.model = create_horizon_model(self.model_type, self.num_classes)
            self.model.to(self.device)
            
            # Optimiseur
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate,
                weight_decay=1e-4
            )
            
            # Scheduleur
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.num_epochs
            )
            
            # Fonction de perte
            self.criterion = HorizonLoss(
                focal_alpha=1.0,
                focal_gamma=2.0,
                dice_weight=0.3
            )
            
        elif self.model_type in ['random_forest', 'svm']:
            # Modèles traditionnels
            self.model = HorizonClassifier(self.model_type)
        
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")
    
    def prepare_data(self, images_dir: str, annotations_dir: str, val_split: float = 0.2):
        """Préparer les données d'entraînement et de validation"""
        
        # Créer le dataset complet
        full_dataset = HorizonDataset(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            augment=True,
            extract_features=True
        )
        
        print(f"Dataset total: {len(full_dataset)} échantillons")
        
        if len(full_dataset) == 0:
            raise ValueError("Aucun échantillon trouvé dans le dataset")
        
        # Division train/validation
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Entraînement: {len(self.train_dataset)} échantillons")
        print(f"Validation: {len(self.val_dataset)} échantillons")
        
        # Créer les dataloaders pour les modèles deep learning
        if self.model_type in ['cnn', 'unet', 'hybrid']:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                collate_fn=self._collate_fn
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                collate_fn=self._collate_fn
            )
    
    def _collate_fn(self, batch):
        """Fonction de regroupement personnalisée pour les batches - format ligne detection"""
        images = []
        target_masks = []
        
        for sample in batch:
            if sample is None:
                continue
                
            image = sample['image']
            target_mask = sample['target_mask']
            
            # Convertir l'image en tensor
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            images.append(image_tensor)
            
            # Convertir le masque cible (0=fond, 1=horizon ligne)
            target_tensor = torch.from_numpy(target_mask.astype(np.int64))
            target_masks.append(target_tensor)
        
        if not images:
            return None
        
        # Empiler les tenseurs
        images_batch = torch.stack(images)
        targets_batch = torch.stack(target_masks)
        
        return {
            'images': images_batch,
            'targets': targets_batch
        }
    
    def train_deep_learning_model(self):
        """Entraîner un modèle deep learning"""
        
        print(f"Début de l'entraînement du modèle {self.model_type}...")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            print(f"\nÉpoque {epoch + 1}/{self.num_epochs}")
            
            # Phase d'entraînement
            train_loss, train_acc = self._train_epoch()
            
            # Phase de validation
            val_loss, val_acc = self._validate_epoch()
            
            # Mettre à jour le scheduler
            self.scheduler.step()
            
            # Sauvegarder l'historique
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Affichage des métriques
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(f'best_{self.model_type}_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping après {epoch + 1} époques")
                    break
        
        print(f"Entraînement terminé. Meilleure validation loss: {best_val_loss:.4f}")
    
    def _train_epoch(self):
        """Entraîner une époque"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Entraînement")
        for batch in pbar:
            if batch is None:
                continue
            
            images = batch['images'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass - utiliser U-Net pour segmentation ligne/fond
            outputs = self.model(images)
            
            # Calculer la perte
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistiques
            total_loss += loss.item()
            
            # Calculer l'accuracy
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == targets).float().sum().item()
            total += targets.numel()
            
            # Mettre à jour la barre de progression
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/total:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self):
        """Valider une époque"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                if batch is None:
                    continue
                
                images = batch['images'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculer la perte
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculer l'accuracy
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == targets).float().sum().item()
                total += targets.numel()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train_traditional_model(self, images_dir: str, annotations_dir: str):
        """Entraîner un modèle traditionnel (Random Forest, SVM)"""
        
        print(f"Entraînement du modèle {self.model_type}...")
        
        # Créer le dataset pour extraire les caractéristiques
        dataset = HorizonDataset(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            extract_features=True,
            augment=False
        )
        
        # Collecter toutes les caractéristiques et labels
        features_list = []
        labels_list = []
        
        for i in tqdm(range(len(dataset)), desc="Extraction des caractéristiques"):
            sample = dataset[i]
            
            if 'features' in sample:
                for label, features in sample['features'].items():
                    features_list.append(features)
                    labels_list.append(label)
        
        if not features_list:
            raise ValueError("Aucune caractéristique extraite")
        
        # Convertir en arrays numpy
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"Données collectées: {X.shape[0]} échantillons, {X.shape[1]} caractéristiques")
        print(f"Classes: {np.unique(y)}")
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entraîner le modèle
        self.model.train(X_train, y_train)
        
        # Évaluation sur le test
        y_pred = self.model.predict(X_test)
        
        print("\nRapport de classification (test):")
        print(classification_report(y_test, y_pred))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, np.unique(y))
        
        return X_test, y_test, y_pred
    
    def plot_confusion_matrix(self, cm, classes):
        """Afficher la matrice de confusion"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Matrice de confusion')
        plt.ylabel('Vraie classe')
        plt.xlabel('Classe prédite')
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """Afficher l'historique d'entraînement"""
        if not self.train_losses:
            print("Aucun historique d'entraînement disponible")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Perte
        ax1.plot(self.train_losses, label='Entraînement')
        ax1.plot(self.val_losses, label='Validation')
        ax1.set_title('Évolution de la perte')
        ax1.set_xlabel('Époque')
        ax1.set_ylabel('Perte')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.train_accuracies, label='Entraînement')
        ax2.plot(self.val_accuracies, label='Validation')
        ax2.set_title('Évolution de la précision')
        ax2.set_xlabel('Époque')
        ax2.set_ylabel('Précision')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename: str):
        """Sauvegarder le modèle"""
        if self.model_type in ['cnn', 'unet', 'hybrid']:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_type': self.model_type,
                'num_classes': self.num_classes,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
            }, filename)
        elif self.model_type in ['random_forest', 'svm']:
            self.model.save(filename)
        
        print(f"Modèle sauvegardé: {filename}")
    
    def train(self, images_dir: str, annotations_dir: str):
        """Méthode principale d'entraînement"""
        start_time = time.time()
        
        # Configurer le modèle
        self.setup_model()
        
        if self.model_type in ['cnn', 'unet', 'hybrid']:
            # Préparer les données
            self.prepare_data(images_dir, annotations_dir)
            
            # Entraîner le modèle deep learning
            self.train_deep_learning_model()
            
            # Afficher l'historique
            self.plot_training_history()
        
        elif self.model_type in ['random_forest', 'svm']:
            # Entraîner le modèle traditionnel
            self.train_traditional_model(images_dir, annotations_dir)
        
        training_time = time.time() - start_time
        print(f"\nTemps d'entraînement total: {training_time/60:.2f} minutes")


def main():
    parser = argparse.ArgumentParser(description="Entraîner un modèle de détection d'horizons")
    parser.add_argument("--images_dir", type=str, required=True,
                       help="Répertoire des images")
    parser.add_argument("--annotations_dir", type=str, required=True,
                       help="Répertoire des annotations JSON")
    parser.add_argument("--model_type", type=str, default="hybrid",
                       choices=['cnn', 'unet', 'hybrid', 'random_forest', 'svm'],
                       help="Type de modèle à entraîner")
    parser.add_argument("--num_classes", type=int, default=3,
                       help="Nombre de classes")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Taille du batch")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Nombre d'époques")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="Taux d'apprentissage")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device à utiliser (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    # Créer l'entraîneur
    trainer = HorizonTrainer(
        model_type=args.model_type,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device
    )
    
    # Lancer l'entraînement
    trainer.train(args.images_dir, args.annotations_dir)


if __name__ == "__main__":
    main()