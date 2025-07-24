#!/usr/bin/env python3
"""
Quick evaluation launcher for soil horizon segmentation model
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch evaluation with trained model"""
    
    # Check if trained model exists
    model_path = Path("checkpoints/best_model.pth")
    if not model_path.exists():
        print("Erreur: Modèle entraîné non trouvé. Lancez d'abord l'entraînement avec train_horizon_model.py")
        return
    
    # Evaluation parameters
    params = [
        sys.executable, "src/evaluate_horizon_segmentation.py",
        "--model-path", str(model_path),
        "--data-dir", "data/processed",
        "--batch-size", "2",
        "--num-classes", "8",
        "--backbone", "resnet50",
        "--image-size", "512", "512",
        "--save-dir", "evaluation_results",
        "--val-split", "0.2",
        "--device", "cuda"  # Force GPU usage
    ]
    
    print("=== Lancement de l'évaluation du modèle de segmentation des horizons ===")
    print(f"Modèle utilisé: {model_path}")
    print(f"Commande: {' '.join(params)}")
    
    # Create evaluation results directory
    Path("evaluation_results").mkdir(exist_ok=True)
    
    # Run evaluation
    try:
        subprocess.run(params, check=True)
        print("\n=== Évaluation terminée avec succès ===")
        print("Consultez le dossier 'evaluation_results' pour les résultats détaillés.")
    except subprocess.CalledProcessError as e:
        print(f"\n=== Erreur lors de l'évaluation: {e} ===")
    except KeyboardInterrupt:
        print("\n=== Évaluation interrompue par l'utilisateur ===")

if __name__ == "__main__":
    main()