#!/usr/bin/env python3
"""
Quick training launcher for soil horizon segmentation model
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch training with optimized parameters"""
    
    # Training parameters for U-Net regression
    params = [
        sys.executable, "src/train_horizon_segmentation.py",
        "--data-dir", "data/processed",
        "--batch-size", "4",  # Reduced for memory constraints
        "--epochs", "50",
        "--lr", "0.001",      # Standard learning rate for U-Net
        "--max-horizons", "7",
        "--val-split", "0.2",
        "--save-dir", "checkpoints",
        "--num-workers", "0",  # Disable multiprocessing for Windows
        "--device", "cuda"    # Force GPU usage
    ]
    
    print("=== Lancement de l'entraînement du modèle de segmentation des horizons ===")
    print(f"Commande: {' '.join(params)}")
    
    # Create checkpoints directory
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Run training
    try:
        subprocess.run(params, check=True)
        print("\n=== Entraînement terminé avec succès ===")
    except subprocess.CalledProcessError as e:
        print(f"\n=== Erreur lors de l'entraînement: {e} ===")
    except KeyboardInterrupt:
        print("\n=== Entraînement interrompu par l'utilisateur ===")

if __name__ == "__main__":
    main()