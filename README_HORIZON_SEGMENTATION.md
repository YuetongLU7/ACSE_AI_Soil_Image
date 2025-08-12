# Segmentation des Horizons de Sol avec DeepLabV3

## Aperçu

Ce module implémente un système de segmentation automatique des horizons de sol utilisant DeepLabV3 avec supervision. Le système utilise les masques d'horizons générés à partir des données Excel pour entraîner un modèle de deep learning.

## Structure des Fichiers

```
src/
├── horizon_segmentation_model.py     # Modèle DeepLabV3 et dataset
├── train_horizon_segmentation.py     # Script d'entraînement
└── evaluate_horizon_segmentation.py  # Script d'évaluation

# Scripts de lancement rapide
train_horizon_model.py                # Lancement de l'entraînement
evaluate_horizon_model.py            # Lancement de l'évaluation
```

## Installation des Dépendances

```bash
pip install -r requirements.txt
```

Dépendances principales ajoutées :
- `albumentations` : Augmentation de données
- `tqdm` : Barres de progression
- `seaborn` : Visualisations
- `scikit-learn` : Métriques d'évaluation

## Utilisation

### 1. Préparation des Données

Assurez-vous que les données ont été prétraitées et que les masques d'horizons ont été générés :

```bash
python src/horizon_label_generator.py
```

Cela créera les fichiers `*_horizon_mask.png` dans `data/processed/`.

### 2. Entraînement du Modèle

#### Méthode Simple (Recommandée)
```bash
python train_horizon_model.py
```

#### Méthode Avancée
```bash
python src/train_horizon_segmentation.py \
    --data-dir data/processed \
    --batch-size 4 \
    --epochs 50 \
    --lr 0.0001 \
    --backbone resnet50 \
    --save-dir checkpoints
```

### 3. Évaluation du Modèle

#### Méthode Simple
```bash
python evaluate_horizon_model.py
```

#### Méthode Avancée
```bash
python src/evaluate_horizon_segmentation.py \
    --model-path checkpoints/best_model.pth \
    --data-dir data/processed \
    --save-dir evaluation_results
```

#### Prédiction sur Image Unique
```bash
python src/evaluate_horizon_segmentation.py \
    --model-path checkpoints/best_model.pth \
    --single-image path/to/image.png
```

## Architecture du Modèle

- **Backbone** : ResNet-50 ou ResNet-101 pré-entraîné sur ImageNet
- **Décodeur** : DeepLabV3 avec ASPP (Atrous Spatial Pyramid Pooling)
- **Classes** : 8 classes maximum (arrière-plan + 7 horizons)
- **Fonction de Perte** : Combinaison CrossEntropy + Dice Loss

## Caractéristiques

### Modèle (`HorizonSegmentationModel`)
- Architecture DeepLabV3 avec backbone configurable
- Support pour modèles pré-entraînés
- Tête de classification personnalisée

### Dataset (`SoilHorizonDataset`)
- Chargement automatique des images et masques
- Support pour jusqu'à 7 horizons par image
- Validation automatique des fichiers requis

### Entraînement
- Calcul automatique des poids de classes (gestion du déséquilibre)
- Augmentation de données avec Albumentations
- Validation croisée avec métriques IoU
- Sauvegarde du meilleur modèle

### Évaluation
- Métriques complètes : Pixel Accuracy, Mean IoU, Class IoU
- Matrice de confusion
- Visualisations des prédictions
- Export des résultats en JSON

## Métriques d'Évaluation

- **Pixel Accuracy** : Pourcentage de pixels correctement classifiés
- **Mean IoU** : Intersection over Union moyenne sur toutes les classes
- **Class IoU** : IoU pour chaque classe d'horizon
- **Mean Accuracy** : Précision moyenne par classe

## Paramètres Recommandés

### Pour Entraînement Initial
```bash
--batch-size 4          # Ajustez selon la mémoire GPU
--lr 0.0001             # Taux d'apprentissage conservateur
--epochs 50             # Nombre d'époques suffisant
--image-size 512 512    # Résolution équilibrée
```

### Pour Fine-tuning
```bash
--lr 0.00001            # Taux plus bas
--backbone resnet101    # Modèle plus complexe
--batch-size 2          # Batch plus petit pour modèle plus grand
```

## Résultats Attendus

Avec 146 échantillons d'entraînement, vous devriez obtenir :
- **Pixel Accuracy** > 85%
- **Mean IoU** > 0.6
- **Convergence** en 30-50 époques

## Dépannage

### Erreurs de Mémoire
- Réduisez `batch_size` à 2 ou 1
- Utilisez `--image-size 256 256`
- Passez à `backbone resnet50`

### Pas de Données Trouvées
```bash
# Vérifiez que les masques existent
ls data/processed/*_horizon_mask.png

# Re-générez si nécessaire
python src/horizon_label_generator.py
```

### Performance Faible
- Augmentez le nombre d'époques
- Ajustez le taux d'apprentissage
- Vérifiez la qualité des données d'entraînement

## Fichiers Générés

### Entraînement
```
checkpoints/
├── best_model.pth              # Meilleur modèle (IoU)
├── checkpoint_epoch_X.pth      # Checkpoints par époque
├── training_curves.png         # Courbes d'entraînement
└── training_history.json       # Historique détaillé
```

### Évaluation
```
evaluation_results/
├── evaluation_metrics.json     # Métriques finales
├── confusion_matrix.png        # Matrice de confusion
└── sample_predictions/         # Échantillons de prédictions
    ├── sample1_prediction.png
    └── ...
```

## Intégration Future

Ce modèle peut être intégré dans le pipeline principal pour :
1. Segmentation automatique des nouveaux échantillons
2. Validation des annotations manuelles
3. Analyse quantitative des horizons
4. Génération de rapports automatiques