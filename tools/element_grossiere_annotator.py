#!/usr/bin/env python3
"""
Element Grossiere annotation tool
Outil d'annotation pour les éléments grossiers du sol
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import random

class ElementGrossiereAnnotator:
    """Outil d'annotation pour les éléments grossiers"""
    
    def __init__(self):
        self.classes = {
            0: {'name': 'arriere_plan', 'color': (0, 0, 0), 'description': 'Background soil'},
            1: {'name': 'pierres', 'color': (255, 0, 0), 'description': 'Stones and gravels'},      
            2: {'name': 'racines', 'color': (0, 255, 0), 'description': 'Plant roots'},       
            3: {'name': 'debris_organiques', 'color': (0, 0, 255), 'description': 'Organic debris'}
        }
        
    def prepare_images_for_annotation(self, processed_dir: str, output_dir: str, num_samples: int = 30):
        """Préparer les images pour l'annotation avec Labelme"""
        processed_path = Path(processed_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Trouver toutes les images traitées
        processed_images = list(processed_path.glob("*_processed.png"))
        
        if not processed_images:
            print("Aucune image traitée trouvée dans", processed_dir)
            return
        
        # Sélectionner un échantillon représentatif
        selected_images = self.select_representative_images(processed_images, num_samples)
        
        # Créer le guide d'annotation
        annotation_guide = {
            'classes': self.classes,
            'annotation_rules': {
                'pierres': {
                    'description': 'Particules minérales dures (cailloux, graviers)',
                    'characteristics': [
                        'Forme angulaire ou arrondie',
                        'Couleur grise, blanche ou brune claire',
                        'Surface lisse ou rugueuse',
                        'Pas de ramifications'
                    ],
                    'size_minimum': 'Diamètre > 2mm',
                    'examples': 'Quartz, calcaire, granite fragmenté'
                },
                'racines': {
                    'description': 'Systèmes racinaires de plantes',
                    'characteristics': [
                        'Forme fibreuse et ramifiée',
                        'Couleur brune à noire',
                        'Structure continue et souple',
                        'Ramifications visibles'
                    ],
                    'size_minimum': 'Diamètre > 2mm',
                    'examples': 'Racines principales, radicelles épaisses'
                },
                'debris_organiques': {
                    'description': 'Matière organique décomposée',
                    'characteristics': [
                        'Forme irrégulière',
                        'Couleur brun foncé à noire',
                        'Texture poreuse ou fibreuse',
                        'Structure fragile'
                    ],
                    'size_minimum': 'Surface > 4mm²',
                    'examples': 'Feuilles mortes, bois décomposé, écorce'
                }
            },
            'workflow': [
                '1. Ouvrir Labelme: labelme',
                '2. Ouvrir le dossier d\'images',
                '3. Créer les polygones pour chaque élément visible',
                '4. Utiliser les labels: pierres, racines, debris_organiques',
                '5. Priorité: éléments les plus grands et évidents d\'abord',
                '6. Ignorer les éléments < 2mm',
                '7. Sauvegarder en format JSON'
            ]
        }
        
        # Sauvegarder le guide
        with open(output_path / "guide_annotation.json", 'w', encoding='utf-8') as f:
            json.dump(annotation_guide, f, indent=2, ensure_ascii=False)
        
        # Copier et redimensionner les images sélectionnées
        annotation_tasks = []
        
        for i, img_path in enumerate(selected_images):
            # Nom de fichier simplifié
            output_img_name = f"sample_{i+1:03d}.png"
            output_img_path = output_path / output_img_name
            
            # Charger et redimensionner l'image si nécessaire
            image = cv2.imread(str(img_path))
            if image is not None:
                h, w = image.shape[:2]
                
                # Redimensionner si trop grande (pour faciliter l'annotation)
                if max(h, w) > 1200:
                    scale = 1200 / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    image = cv2.resize(image, (new_w, new_h))
                    print(f"Image {img_path.name} redimensionnée de {w}x{h} à {new_w}x{new_h}")
                
                cv2.imwrite(str(output_img_path), image)
                
                annotation_tasks.append({
                    'sample_id': i + 1,
                    'image_file': output_img_name,
                    'original_source': str(img_path),
                    'status': 'à_annoter'
                })
        
        # Sauvegarder la liste des tâches
        with open(output_path / "taches_annotation.json", 'w', encoding='utf-8') as f:
            json.dump(annotation_tasks, f, indent=2, ensure_ascii=False)
        
        print(f"=== Préparation terminée ===")
        print(f"Nombre d'images sélectionnées: {len(selected_images)}")
        print(f"Dossier d'annotation: {output_path}")
        print(f"Guide d'annotation: {output_path}/guide_annotation.json")
        
        print(f"\n=== Instructions pour Labelme ===")
        print(f"1. Installer Labelme: pip install labelme")
        print(f"2. Lancer Labelme: labelme")
        print(f"3. Ouvrir le dossier: {output_path}")
        print(f"4. Utiliser les labels: pierres, racines, debris_organiques")
        print(f"5. Après annotation, lancer la conversion:")
        print(f"   python tools/element_grossiere_annotator.py --mode convert --input {output_path} --output data/element_grossiere_dataset")
        
        return output_path
    
    def select_representative_images(self, image_paths: List[Path], num_samples: int) -> List[Path]:
        """Sélectionner un échantillon représentatif d'images"""
        if len(image_paths) <= num_samples:
            return image_paths
        
        # Mélanger et sélectionner
        shuffled = list(image_paths)
        random.seed(42)  # Pour la reproductibilité
        random.shuffle(shuffled)
        
        return shuffled[:num_samples]
    
    def convert_labelme_to_dataset(self, annotation_dir: str, output_dir: str):
        """Convertir les annotations Labelme en dataset d'entraînement"""
        annotation_path = Path(annotation_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Créer les sous-dossiers
        (output_path / "images").mkdir(exist_ok=True)
        (output_path / "masks").mkdir(exist_ok=True)
        
        # Trouver tous les fichiers JSON d'annotation
        json_files = [f for f in annotation_path.glob("*.json") 
                     if f.name not in ['guide_annotation.json', 'taches_annotation.json']]
        
        if not json_files:
            print("Aucun fichier d'annotation trouvé")
            return
        
        dataset_info = {
            'dataset_name': 'element_grossiere_soil',
            'classes': self.classes,
            'images': [],
            'statistics': {
                'total_images': 0,
                'total_annotations': 0,
                'class_counts': {name: 0 for name in [cls['name'] for cls in self.classes.values()]}
            }
        }
        
        processed_count = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    labelme_data = json.load(f)
                
                # Trouver l'image correspondante
                image_name = labelme_data['imagePath']
                image_path = annotation_path / image_name
                
                if not image_path.exists():
                    print(f"Image non trouvée: {image_path}")
                    continue
                
                # Charger l'image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                
                h, w = image.shape[:2]
                
                # Créer le masque de segmentation
                mask = np.zeros((h, w), dtype=np.uint8)
                annotation_count = 0
                
                for shape in labelme_data['shapes']:
                    label = shape['label']
                    points = shape['points']
                    
                    # Obtenir l'ID de classe
                    class_id = self.get_class_id(label)
                    if class_id is None:
                        print(f"Label inconnu: {label}")
                        continue
                    
                    # Créer le polygone
                    pts = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], class_id)
                    
                    annotation_count += 1
                    dataset_info['statistics']['class_counts'][label] += 1
                
                # Sauvegarder l'image et le masque
                base_name = f"sample_{processed_count + 1:03d}"
                
                img_filename = f"{base_name}.png"
                mask_filename = f"{base_name}_mask.png"
                
                cv2.imwrite(str(output_path / "images" / img_filename), image)
                cv2.imwrite(str(output_path / "masks" / mask_filename), mask)
                
                dataset_info['images'].append({
                    'id': processed_count + 1,
                    'filename': img_filename,
                    'mask_filename': mask_filename,
                    'width': w,
                    'height': h,
                    'annotation_count': annotation_count,
                    'source': str(json_file)
                })
                
                dataset_info['statistics']['total_annotations'] += annotation_count
                processed_count += 1
                
            except Exception as e:
                print(f"Erreur lors du traitement de {json_file}: {e}")
        
        dataset_info['statistics']['total_images'] = processed_count
        
        # Sauvegarder les informations du dataset
        with open(output_path / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"=== Conversion terminée ===")
        print(f"Images traitées: {processed_count}")
        print(f"Total annotations: {dataset_info['statistics']['total_annotations']}")
        print(f"Dataset sauvegardé dans: {output_path}")
        
        # Afficher les statistiques par classe
        print(f"\nStatistiques par classe:")
        for class_name, count in dataset_info['statistics']['class_counts'].items():
            if count > 0:
                print(f"  {class_name}: {count} instances")
        
        return output_path
    
    def get_class_id(self, label_name: str) -> int:
        """Obtenir l'ID de classe à partir du nom"""
        for class_id, info in self.classes.items():
            if info['name'] == label_name:
                return class_id
        return None
    
    def analyze_dataset(self, dataset_dir: str):
        """Analyser les statistiques du dataset"""
        dataset_path = Path(dataset_dir)
        info_file = dataset_path / "dataset_info.json"
        
        if not info_file.exists():
            print("Fichier dataset_info.json non trouvé")
            return
        
        with open(info_file, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
        
        print("=== Analyse du Dataset Element Grossiere ===")
        stats = dataset_info['statistics']
        print(f"Nombre total d'images: {stats['total_images']}")
        print(f"Nombre total d'annotations: {stats['total_annotations']}")
        
        if stats['total_images'] > 0:
            avg_annotations = stats['total_annotations'] / stats['total_images']
            print(f"Moyenne d'annotations par image: {avg_annotations:.1f}")
        
        print(f"\nRépartition par classe:")
        for class_name, count in stats['class_counts'].items():
            if count > 0:
                percentage = (count / stats['total_annotations']) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        # Vérifier l'équilibre des classes
        counts = [c for c in stats['class_counts'].values() if c > 0]
        if counts:
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"\nDéséquilibre des classes: {imbalance_ratio:.1f}:1")
            if imbalance_ratio > 10:
                print("⚠️  Déséquilibre important détecté. Considérer l'augmentation de données.")


def main():
    parser = argparse.ArgumentParser(description='Outil d\'annotation Element Grossiere')
    parser.add_argument('--mode', choices=['prepare', 'convert', 'analyze'], required=True,
                       help='Mode d\'opération')
    parser.add_argument('--input', required=True, help='Dossier d\'entrée')
    parser.add_argument('--output', required=True, help='Dossier de sortie')
    parser.add_argument('--samples', type=int, default=30, 
                       help='Nombre d\'échantillons à annoter (mode prepare)')
    
    args = parser.parse_args()
    
    annotator = ElementGrossiereAnnotator()
    
    if args.mode == 'prepare':
        annotator.prepare_images_for_annotation(args.input, args.output, args.samples)
        
    elif args.mode == 'convert':
        annotator.convert_labelme_to_dataset(args.input, args.output)
        
    elif args.mode == 'analyze':
        annotator.analyze_dataset(args.input)


if __name__ == "__main__":
    main()