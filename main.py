#!/usr/bin/env python3
"""
土壤图像预处理主程序
"""

import argparse
import sys
from pathlib import Path
from src.preprocessing import SoilImagePreprocessor

def main():
    parser = argparse.ArgumentParser(description=' Prétraitement des images de sol')
    parser.add_argument('--input', '-i', required=True, 
                       help='Chemin de l\'image ou du répertoire d\'entrée')
    parser.add_argument('--output', '-o', required=True,
                       help='Chemin du répertoire de sortie')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='Chemin du fichier de configuration')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Mode de traitement par lots')
    parser.add_argument('--create-dataset', action='store_true',
                       help='Créer un ensemble de données d\'entraînement')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Proportion de l\'ensemble d\'entraînement (pour la création de l\'ensemble de données)')
    parser.add_argument('--no-clear', action='store_true',
                       help='Ne pas vider le répertoire de sortie')
    parser.add_argument('--incremental', action='store_true',
                       help='Mode de traitement incrémentiel (ne traiter que les nouvelles images)')
    parser.add_argument('--disable-quality-filter', action='store_true',
                       help='Désactiver le filtrage de qualité')

    args = parser.parse_args()
    
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Erreur: le chemin d'entrée n'existe pas: {args.input}")
        sys.exit(1)
    
    # 创建预处理器
    try:
        preprocessor = SoilImagePreprocessor(args.config)
        
        # 根据命令行参数调整质量过滤设置
        if args.disable_quality_filter:
            preprocessor.quality_assessor.enable_filtering = False
            print("Filtrage de qualité désactivé")

        print(f"Utilisation du fichier de configuration: {args.config}")
    except Exception as e:
        print(f"Erreur: impossible de charger le fichier de configuration: {e}")
        sys.exit(1)
    
    # 处理图像
    if args.batch or input_path.is_dir():
        clear_output = not args.no_clear
        if clear_output:
            print("Début du traitement par lots (vider le répertoire de sortie)...")
        elif args.incremental:
            print("Début du traitement par lots incrémentiel...")
        else:
            print("Début du traitement par lots (conserver les fichiers existants)...")

        results = preprocessor.process_batch(
            str(input_path), 
            args.output, 
            clear_output=clear_output,
            incremental=args.incremental
        )
        print(f"Traitement par lots terminé, {len(results)} images traitées")

        # 创建训练数据集
        if args.create_dataset:
            dataset_dir = Path(args.output) / "dataset"
            preprocessor.create_training_dataset(results, str(dataset_dir), args.train_ratio)
            
    else:
        print("Début du traitement d'une seule image...")
        result = preprocessor.process_single_image(str(input_path), args.output)
        print(f"Traitement de l'image terminé: {result['image_name']}")

        if result['ruler_info'] and result['ruler_info']['ruler_detected']:
            print(f"Détection du mètre réussie, échelle: {result['ruler_info']['scale_ratio']:.2f} pixels/cm")
        else:
            print("Aucun mètre détecté")

        print(f"Détection de {len(result['detected_objects'])} objets")

if __name__ == "__main__":
    main()