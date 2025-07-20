import pandas as pd
import numpy as np
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class HorizonLabelGenerator:
    """Generator for soil horizon labels from Excel description data"""
    
    def __init__(self, excel_path: str, processed_data_dir: str):
        """
        Initialize horizon label generator
        
        Args:
            excel_path: Path to Excel file with horizon descriptions
            processed_data_dir: Directory containing preprocessed images and metadata
        """
        self.excel_path = excel_path
        self.processed_data_dir = Path(processed_data_dir)
        self.horizon_data = None
        
    def parse_excel_data(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Parse Excel file to extract horizon depth information
        
        Returns:
            Dict: {profile_name: [(start_cm, end_cm), ...]}
        """
        try:
            df = pd.read_excel(self.excel_path)
            print(f"Lecture Excel réussie: {len(df)} lignes de données")
            
            # Check required columns
            required_cols = ['no_profil', 'no_horizon', 'prof_inf_moy']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Colonnes manquantes dans Excel: {missing_cols}")
            
            horizon_data = {}
            
            # Group by profile and process each
            for profile_name in df['no_profil'].unique():
                if pd.isna(profile_name):
                    continue
                
                # Get all horizons for this profile
                profile_rows = df[df['no_profil'] == profile_name].copy()
                profile_rows = profile_rows.sort_values('no_horizon')
                
                # Extract depth values (prof_inf_moy = bottom depth of each horizon)
                depths = profile_rows['prof_inf_moy'].dropna().tolist()
                
                if not depths:
                    print(f"  Ignoré {profile_name}: aucune donnée de profondeur")
                    continue
                
                # Convert to horizon ranges: (start_cm, end_cm)
                horizons = []
                prev_depth = 0
                
                for depth in depths:
                    if depth > prev_depth:
                        horizons.append((prev_depth, int(depth)))
                        prev_depth = int(depth)
                    else:
                        print(f"  Avertissement {profile_name}: profondeur non croissante ignorée {depth}")
                
                if horizons:
                    horizon_data[str(profile_name)] = horizons
                    print(f"  Profil {profile_name}: {len(horizons)} horizons, profondeurs {horizons}")
            
            self.horizon_data = horizon_data
            print(f"\\nAnalyse Excel terminée: {len(horizon_data)} profils traités")
            return horizon_data
            
        except Exception as e:
            print(f"Erreur lors de l'analyse Excel: {e}")
            return {}
    
    def cm_to_pixel_coordinates(self, horizon_depths: List[Tuple[int, int]], 
                               ruler_info: Dict) -> List[Tuple[int, int]]:
        """
        Convert cm depths to pixel coordinates using ruler information
        
        Args:
            horizon_depths: List of (start_cm, end_cm) tuples
            ruler_info: Ruler detection information from metadata
            
        Returns:
            List of (start_y, end_y) pixel coordinates
        """
        if not ruler_info or not ruler_info.get('ruler_detected'):
            return []
        
        scale_ratio = ruler_info['scale_ratio']  # pixels/cm
        top_digit_y = ruler_info.get('top_digit_y', 0)
        top_digit_value = ruler_info.get('top_digit_value', 10)
        
        # Calculate 0cm reference point
        zero_depth_y = top_digit_y - (top_digit_value * scale_ratio)
        
        pixel_horizons = []
        for start_cm, end_cm in horizon_depths:
            start_y = int(zero_depth_y + (start_cm * scale_ratio))
            end_y = int(zero_depth_y + (end_cm * scale_ratio))
            pixel_horizons.append((start_y, end_y))
        
        return pixel_horizons
    
    def create_horizon_mask(self, image_shape: Tuple[int, int], 
                           pixel_horizons: List[Tuple[int, int]], 
                           soil_mask: np.ndarray) -> np.ndarray:
        """
        Create horizon segmentation mask
        
        Args:
            image_shape: (height, width) of the image
            pixel_horizons: List of (start_y, end_y) pixel coordinates
            soil_mask: Binary soil region mask
            
        Returns:
            np.ndarray: Horizon mask with values 0=background, 1=horizon1, 2=horizon2, etc.
        """
        h, w = image_shape
        horizon_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert soil mask to binary
        soil_binary = (soil_mask > 0).astype(np.uint8)
        
        for horizon_id, (start_y, end_y) in enumerate(pixel_horizons, 1):
            # Clamp coordinates to image bounds
            start_y = max(0, min(start_y, h-1))
            end_y = max(0, min(end_y, h-1))
            
            if start_y >= end_y:
                continue
            
            # Create horizon region
            horizon_region = np.zeros((h, w), dtype=np.uint8)
            horizon_region[start_y:end_y, :] = horizon_id
            
            # Only label within soil regions
            valid_region = cv2.bitwise_and(horizon_region, soil_binary)
            horizon_mask[valid_region > 0] = horizon_id
        
        return horizon_mask
    
    def process_single_profile(self, profile_name: str) -> bool:
        """
        Process a single profile to generate horizon labels
        
        Args:
            profile_name: Name of the profile to process
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.horizon_data is None:
            print("Erreur: données Excel non analysées")
            return False
        
        if profile_name not in self.horizon_data:
            print(f"Profil {profile_name} non trouvé dans les données Excel")
            return False
        
        # Check for required files
        metadata_file = self.processed_data_dir / f"{profile_name}_metadata.json"
        soil_mask_file = self.processed_data_dir / f"{profile_name}_soil_mask.png"
        processed_image_file = self.processed_data_dir / f"{profile_name}_processed.png"
        
        missing_files = [f for f in [metadata_file, soil_mask_file, processed_image_file] 
                        if not f.exists()]
        if missing_files:
            print(f"Fichiers manquants pour {profile_name}: {[f.name for f in missing_files]}")
            return False
        
        try:
            # Load metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Load soil mask
            soil_mask = cv2.imread(str(soil_mask_file), cv2.IMREAD_GRAYSCALE)
            
            # Load processed image for dimensions
            processed_image = cv2.imread(str(processed_image_file))
            
            if soil_mask is None or processed_image is None:
                print(f"Impossible de charger les images pour {profile_name}")
                return False
            
            # Get horizon depths for this profile
            horizon_depths = self.horizon_data[profile_name]
            
            # Convert to pixel coordinates
            ruler_info = metadata.get('ruler_info')
            pixel_horizons = self.cm_to_pixel_coordinates(horizon_depths, ruler_info)
            
            if not pixel_horizons:
                print(f"Impossible de convertir les coordonnées pour {profile_name}")
                return False
            
            # Create horizon mask
            horizon_mask = self.create_horizon_mask(
                processed_image.shape[:2], 
                pixel_horizons, 
                soil_mask
            )
            
            # Save horizon mask
            horizon_mask_file = self.processed_data_dir / f"{profile_name}_horizon_mask.png"
            cv2.imwrite(str(horizon_mask_file), horizon_mask)
            
            # Update metadata with horizon information
            metadata['horizon_info'] = {
                'horizon_depths_cm': horizon_depths,
                'pixel_horizons': pixel_horizons,
                'num_horizons': len(horizon_depths),
                'has_horizon_labels': True
            }
            
            # Save updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            print(f"  ✓ {profile_name}: {len(horizon_depths)} horizons générés")
            return True
            
        except Exception as e:
            print(f"  ✗ {profile_name}: Erreur - {e}")
            return False
    
    def generate_all_labels(self) -> Dict[str, int]:
        """
        Generate horizon labels for all profiles in Excel data
        
        Returns:
            Dict: Processing statistics
        """
        if self.horizon_data is None:
            self.parse_excel_data()
        
        if not self.horizon_data:
            print("Aucune donnée d'horizon à traiter")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        print(f"\\n=== Génération des étiquettes d'horizons ===")
        print(f"Traitement de {len(self.horizon_data)} profils...")
        
        stats = {'total': len(self.horizon_data), 'success': 0, 'failed': 0}
        
        for profile_name in self.horizon_data.keys():
            if self.process_single_profile(profile_name):
                stats['success'] += 1
            else:
                stats['failed'] += 1
        
        print(f"\\n=== Résumé ===")
        print(f"Total: {stats['total']}")
        print(f"Succès: {stats['success']}")
        print(f"Échecs: {stats['failed']}")
        print(f"Taux de réussite: {stats['success']/stats['total']*100:.1f}%")
        
        return stats


def main():
    """Test the horizon label generation"""
    
    # Configuration paths
    excel_path = "/mnt/e/CodeForStudy/Stage/Projet/ACSE_AI_Soil_Image/data/description/hrz_description_photo.xlsx"
    processed_dir = "/mnt/e/CodeForStudy/Stage/Projet/ACSE_AI_Soil_Image/data/processed"
    
    print("=== Test de génération d'étiquettes d'horizons ===")
    
    # Initialize generator
    generator = HorizonLabelGenerator(excel_path, processed_dir)
    
    # Parse Excel data first
    horizon_data = generator.parse_excel_data()
    
    if not horizon_data:
        print("Erreur: Impossible de lire les données Excel")
        return
    
    # Test with single profile first
    test_profile = "F54001r"
    if test_profile in horizon_data:
        print(f"\\n=== Test avec profil {test_profile} ===")
        success = generator.process_single_profile(test_profile)
        if success:
            print(f"Test réussi pour {test_profile}")
        else:
            print(f"Test échoué pour {test_profile}")
    
    # Generate labels for all profiles
    print(f"\\n=== Génération pour tous les profils ===")
    stats = generator.generate_all_labels()


if __name__ == "__main__":
    main()