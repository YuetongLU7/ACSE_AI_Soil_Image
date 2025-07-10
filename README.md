# ACSE_AI_Soil_Image - Prétraitement d'Images de Sol

Ce projet implémente un pipeline de prétraitement automatisé pour les images de sol, incluant la détection de mètre-ruban, la segmentation des zones de sol et l'évaluation de la qualité.

## Objectifs

- Détecter automatiquement les mètres-rubans dans les images de sol
- Calculer le ratio pixels/cm pour l'analyse dimensionnelle
- Segmenter les zones de sol en excluant le ciel, la végétation et les outils
- Évaluer la qualité des images et filtrer les images de faible qualité
- Créer des ensembles de données d'entraînement pour les modèles IA

## Structure du Projet

```
ACSE_AI_Soil_Image/
├── data/
│   ├── raw/                    # Images originales
│   ├── processed/              # Images prétraitées
│   └── training_dataset/       # Ensemble de données d'entraînement
├── src/
│   ├── preprocessing.py        # Pipeline principal de prétraitement
│   ├── ruler_detection.py      # Détection de mètre-ruban avec OCR
│   ├── soil_segmentation.py    # Segmentation des zones de sol
│   ├── quality_assessment.py   # Évaluation de la qualité d'image
│   └── utils.py               # Fonctions utilitaires
├── config/
│   └── config.yaml            # Fichier de configuration
├── tests/
│   ├── src/
│   │   └── test_algorithms.py  # Tests des algorithmes
│   └── README.md              # Guide de test
├── requirements.txt           # Dépendances Python
└── main.py                   # Point d'entrée principal
```

## Technologies Utilisées

- **OpenCV**: Traitement d'images et vision par ordinateur
- **NumPy**: Calculs numériques et manipulation de matrices
- **EasyOCR/Tesseract**: Reconnaissance optique de caractères (OCR)
- **YAML**: Configuration
- **Pathlib**: Gestion des chemins de fichiers
- **JSON**: Stockage des métadonnées

## Installation

### Prérequis
- Python 3.8+
- pip

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Installation d'EasyOCR (recommandé pour de meilleures performances)
```bash
pip install easyocr
```

### Installation de Tesseract OCR (alternative)
- **Windows**: Télécharger depuis [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux**: `sudo apt-get install tesseract-ocr`
- **macOS**: `brew install tesseract`

## Utilisation

### Traitement d'une seule image
```bash
python main.py -i chemin/vers/image.jpg -o dossier/sortie/
```

### Traitement par lots
```bash
python main.py -i data/raw/ -o data/processed/ --batch
```

### Traitement incrémentiel (ne traite que les nouvelles images)
```bash
python main.py -i data/raw/ -o data/processed/ --batch --incremental
```

### Création d'un ensemble de données d'entraînement
```bash
python main.py -i data/raw/ -o data/processed/ --batch --create-dataset --train-ratio 0.8
```

### Options avancées
```bash
python main.py -i data/raw/ -o data/processed/ --batch \
  --incremental \
  --create-dataset \
  --train-ratio 0.8 \
  --disable-quality-filter \
  --no-clear
```

## Paramètres de Configuration

Éditez `config/config.yaml` pour ajuster les paramètres:

```yaml
preprocessing:
  ruler_detection:
    min_length: 100
    max_length: 2000
  soil_segmentation:
    mask_type: 'transparent'  # ou 'black'

quality_assessment:
  enable_filtering: true
  thresholds:
    min_soil_coverage: 0.3
    max_reflection_ratio: 0.15
    min_contrast: 100
    max_shadow_ratio: 0.25
    min_mask_connectivity: 0.7
```

## Fonctionnalités Principales

### 1. Détection de Mètre-Ruban
- **Méthode principale**: Détection OCR des chiffres d'échelle (0, 10, 20, ..., 120cm)
- **Méthode de secours**: Détection morphologique
- **Masquage double**: Zone du mètre + zone des chiffres
- **Calcul automatique**: Ratio pixels/cm

### 2. Segmentation du Sol
- **Méthode d'exclusion**: Élimine les zones non-sol
- **Zones exclues**: Ciel, végétation, mètre-ruban, outils métalliques
- **Algorithmes**: Détection HSV + opérations morphologiques

### 3. Évaluation de la Qualité
- **Métriques**: Couverture sol, reflets, contraste, ombres, connectivité
- **Filtrage automatique**: Sépare les images haute/basse qualité
- **Rapports détaillés**: Statistiques et problèmes identifiés

### 4. Ensemble de Données d'Entraînement
- **Division automatique**: 80% entraînement / 20% validation
- **Structure organisée**: Images et masques séparés
- **Compatible IA**: Format prêt pour U-Net, DeepLab, etc.

## Fichiers de Sortie

Pour chaque image traitée:
```
data/processed/
├── nom_image_processed.png      # Image traitée
├── nom_image_soil_mask.png      # Masque de sol (blanc = sol)
├── nom_image_remove_mask.png    # Masque de suppression
├── nom_image_ruler_detection.png # Visualisation de détection
└── nom_image_metadata.json     # Métadonnées complètes
```

Rapports globaux:
```
├── batch_processing_report.json     # Rapport de traitement par lots
├── quality_assessment_report.json   # Rapport d'évaluation qualité
└── dataset/                         # Ensemble de données d'entraînement
    ├── train/ (80%)
    └── val/ (20%)
```

## Tests

Exécuter les tests d'algorithmes:
```bash
cd tests/src/
python test_algorithms.py
```

## Dépannage

### Problèmes courants

1. **Erreur OCR**: Vérifiez l'installation de EasyOCR/Tesseract
2. **Noms de fichiers chinois**: Le système gère automatiquement l'encodage UTF-8
3. **Mémoire insuffisante**: Réduisez la taille du lot ou utilisez le mode incrémentiel
4. **Aucun mètre détecté**: Vérifiez que les chiffres d'échelle sont visibles

### Logs et débogage
Les messages d'état sont affichés en temps réel. Pour plus de détails, consultez les fichiers JSON de sortie.

## Contribution

Ce projet fait partie d'un stage de recherche. Pour les contributions:
1. Créez une branche pour vos modifications
2. Testez avec le module de test
3. Documentez les changements
4. Soumettez une pull request

## Licence

Projet de stage de recherche - Utilisation académique.

## Contexte Académique

Développé dans le cadre d'un stage de recherche sur l'analyse automatisée des images de sol pour l'agriculture de précision et la science du sol.

---

**Contact**: Pour toute question technique, consultez les issues du dépôt ou la documentation inline dans le code.