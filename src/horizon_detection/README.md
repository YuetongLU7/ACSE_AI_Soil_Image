# Système de Détection d'Horizons de Sol

Ce module implémente un système complet de détection d'horizons de sol basé sur l'apprentissage automatique. Il utilise une approche hybride combinant des caractéristiques de texture et couleur avec des réseaux de neurones convolutifs pour identifier automatiquement les délimitations d'horizons dans les profils de sol.

## 🎯 Objectif

Développer un modèle capable d'apprendre les patterns de texture et couleur des horizons de sol à partir d'annotations manuelles, puis prédire automatiquement les délimitations d'horizons sur de nouvelles images de profils de sol.

## 📁 Structure des Fichiers

```
src/horizon_detection/
├── README.md                    # Ce fichier
├── data_loader.py              # Chargement et préparation des données
├── horizon_model.py            # Architectures des modèles
├── train_horizon_model.py      # Script d'entraînement
└── predict_horizon.py          # Script de prédiction
```

## 🔧 Installation des Dépendances

```bash
pip install torch torchvision
pip install opencv-python
pip install scikit-image
pip install scikit-learn
pip install albumentations
pip install segmentation-models-pytorch
pip install matplotlib seaborn
pip install tqdm joblib
```

## 📊 Format des Données

### Images d'entrée
- Format: JPG, PNG, BMP, TIFF
- Taille recommandée: Minimum 512x512 pixels
- Les images doivent montrer des profils de sol avec des horizons visibles

### Annotations
- Format: JSON (LabelMe)
- Structure attendue:
```json
{
  "version": "5.8.3",
  "shapes": [
    {
      "label": "0",  // Classe d'horizon (0, 1, 2, ...)
      "points": [    // Points du polygone
        [x1, y1],
        [x2, y2],
        ...
      ]
    }
  ]
}
```

## 🚀 Utilisation

### 1. Entraînement d'un Modèle

#### Modèle Hybride (Recommandé)
```bash
python src/horizon_detection/train_horizon_model.py \
    --images_dir data/delimitation_horizons \
    --annotations_dir data/horizon \
    --model_type hybrid \
    --num_classes 3 \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 1e-3
```

#### Modèle U-Net
```bash
python src/horizon_detection/train_horizon_model.py \
    --images_dir data/delimitation_horizons \
    --annotations_dir data/horizon \
    --model_type unet \
    --num_classes 3 \
    --batch_size 4 \
    --num_epochs 50
```

#### Modèle CNN Simple
```bash
python src/horizon_detection/train_horizon_model.py \
    --images_dir data/delimitation_horizons \
    --annotations_dir data/horizon \
    --model_type cnn \
    --num_classes 3 \
    --batch_size 4 \
    --num_epochs 50
```

#### Modèles Traditionnels
```bash
# Random Forest
python src/horizon_detection/train_horizon_model.py \
    --images_dir data/delimitation_horizons \
    --annotations_dir data/horizon \
    --model_type random_forest \
    --num_classes 3

# SVM
python src/horizon_detection/train_horizon_model.py \
    --images_dir data/delimitation_horizons \
    --annotations_dir data/horizon \
    --model_type svm \
    --num_classes 3
```

### 2. Prédiction sur de Nouvelles Images

#### Prédiction sur une Image
```bash
python src/horizon_detection/predict_horizon.py \
    --model_path best_hybrid_model.pth \
    --model_type hybrid \
    --image_path test_image.jpg \
    --output_dir results/
```

#### Prédiction en Lot
```bash
python src/horizon_detection/predict_horizon.py \
    --model_path best_hybrid_model.pth \
    --model_type hybrid \
    --image_dir test_images/ \
    --output_dir results/
```

## 🎯 Types de Modèles Disponibles

### 1. Modèle Hybride (Recommandé)
- **Avantages**: Combine CNN et caractéristiques extraites manuellement
- **Utilisation**: Meilleurs résultats sur des données limitées
- **Caractéristiques**: Texture (LBP, GLCM, Gabor) + Couleur (BGR, HSV, LAB) + Géométrie

### 2. U-Net
- **Avantages**: Architecture éprouvée pour la segmentation
- **Utilisation**: Bon pour la segmentation pixel par pixel
- **Caractéristiques**: Architecture encoder-decoder avec connections skip

### 3. CNN Texture-Couleur
- **Avantages**: Focus sur les caractéristiques de texture et couleur
- **Utilisation**: Analyse des patterns visuels spécifiques aux horizons
- **Caractéristiques**: Branches spécialisées texture et couleur

### 4. Random Forest / SVM
- **Avantages**: Rapides à entraîner, interprétables
- **Utilisation**: Baseline et comparaison
- **Caractéristiques**: Basés uniquement sur les caractéristiques extraites

## 📈 Caractéristiques Extraites

### Texture
- **Local Binary Pattern (LBP)**: Motifs locaux de texture
- **Gray Level Co-occurrence Matrix (GLCM)**: Relations spatiales des pixels
- **Filtres de Gabor**: Réponses directionnelles

### Couleur
- **Espace BGR**: Couleurs de base
- **Espace HSV**: Teinte, saturation, valeur
- **Espace LAB**: Perception visuelle humaine

### Géométrie
- **Aire**: Surface de la région
- **Périmètre**: Contour de la région
- **Excentricité**: Forme de l'ellipse équivalente
- **Solidité**: Compacité de la forme
- **Rapport d'aspect**: Relation largeur/hauteur

## 📊 Évaluation des Modèles

Les modèles sont évalués sur:
- **Accuracy**: Précision globale
- **Loss**: Fonction de perte (Focal + Dice pour segmentation)
- **IoU**: Intersection over Union pour la segmentation
- **Classification Report**: Précision, rappel, F1-score par classe

## 🎨 Visualisation des Résultats

Les scripts génèrent automatiquement:
- **Masques de prédiction**: Segmentation colorée par classe
- **Contours détectés**: Délimitations des horizons
- **Boîtes englobantes**: Régions d'intérêt
- **Statistiques**: Métriques de performance
- **Historique d'entraînement**: Courbes de loss et accuracy

## 📝 Exemples d'Usage

### Test Rapide avec Random Forest
```python
from horizon_model import HorizonClassifier
from data_loader import HorizonDataset

# Créer le dataset
dataset = HorizonDataset(\n    \"data/delimitation_horizons\", \n    \"data/horizon\"\n)\n\n# Entraîner un classificateur rapide\nclassifier = HorizonClassifier('random_forest')\n# ... (voir scripts complets)\n```

### Entraînement Personnalisé
```python\nfrom train_horizon_model import HorizonTrainer\n\ntrainer = HorizonTrainer(\n    model_type='hybrid',\n    num_classes=3,\n    learning_rate=1e-3,\n    batch_size=4\n)\n\ntrainer.train(\n    \"data/delimitation_horizons\",\n    \"data/horizon\"\n)\n```

## ⚙️ Configuration Avancée

### Augmentation de Données
- Retournement horizontal
- Variations de luminosité/contraste
- Changements de teinte/saturation
- Ajout de bruit gaussien
- Flou

### Hyperparamètres Recommandés
- **Batch Size**: 4-8 (selon GPU)
- **Learning Rate**: 1e-3 à 1e-4
- **Epochs**: 50-100
- **Image Size**: 512x512
- **Optimizer**: AdamW avec weight decay

## 🚨 Dépannage

### Erreurs Communes

1. **Manque de mémoire GPU**
   - Réduire batch_size
   - Réduire image_size
   - Utiliser device='cpu'

2. **Pas de convergence**
   - Vérifier les données d'entrée
   - Ajuster le learning rate
   - Augmenter le nombre d'époques

3. **Overfitting**
   - Augmenter l'augmentation de données
   - Réduire la complexité du modèle
   - Ajouter plus de données d'entraînement

### Messages d'Erreur
- `\"Aucun échantillon trouvé\"`: Vérifier les chemins des données
- `\"CUDA out of memory\"`: Réduire batch_size
- `\"Model type not supported\"`: Vérifier model_type

## 📚 Références Scientifiques

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- Focal Loss for Dense Object Detection
- Local Binary Patterns for Texture Classification
- Gray Level Co-occurrence Matrix Analysis

## 🤝 Contribution

Pour améliorer ce système:
1. Ajouter de nouvelles caractéristiques de texture
2. Implémenter d'autres architectures (DeepLab, PSPNet)
3. Optimiser les hyperparamètres
4. Améliorer la visualisation des résultats

## 📞 Support

Pour des questions ou problèmes:
1. Vérifier ce README
2. Consulter les docstrings du code
3. Tester avec les données d'exemple
4. Vérifier les logs d'erreur pour diagnostiquer

---

**Note**: Ce système est conçu spécifiquement pour l'analyse des horizons de sol. Les performances peuvent varier selon la qualité et la diversité des données d'entraînement.