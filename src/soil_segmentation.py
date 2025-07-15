import cv2
import numpy as np
from typing import List, Tuple, Optional
from skimage import feature, segmentation, measure
from scipy import ndimage

class MorphologicalSoilSegmentation:
    """Morphological and texture-based soil segmentation"""
    
    def __init__(self):
        pass
    
    def segment_soil_area(self, image: np.ndarray, ruler_mask: np.ndarray = None, 
                         upper_boundary: int = None, debug: bool = False) -> np.ndarray:
        """
        Segment soil areas based on morphological features
        
        Args:
            image: Input image
            ruler_mask: Ruler mask
            upper_boundary: Upper boundary y-coordinate
            debug: Whether to return debug information
            
        Returns:
            Soil area mask or debug information dictionary
        """
        h, w = image.shape[:2]
        
        # Create initial mask
        soil_mask = np.ones((h, w), dtype=np.uint8) * 255
        debug_masks = {} if debug else None
        
        # 1. Upper boundary filtering
        if upper_boundary is not None and 0 <= upper_boundary < h:
            print(f"Application du masque de limite supérieure: y < {upper_boundary}")
            upper_boundary_mask = np.zeros((h, w), dtype=np.uint8)
            upper_boundary_mask[:upper_boundary, :] = 255
            soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(upper_boundary_mask))
            if debug: debug_masks['upper_boundary'] = upper_boundary_mask
        
        # 2. Detect sky (large uniform regions)
        sky_mask = self._detect_sky_morphological(image)
        soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(sky_mask))
        if debug: debug_masks['sky'] = sky_mask
        
        # 3. Detect vegetation (linear texture features)
        vegetation_mask = self._detect_vegetation_morphological(image)
        soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(vegetation_mask))
        if debug: debug_masks['vegetation'] = vegetation_mask
        
        # 4. Detect ruler
        if ruler_mask is not None:
            soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(ruler_mask))
            if debug: debug_masks['ruler'] = ruler_mask
        
        # 5. Tool detection removed (to avoid soil misclassification)
        
        # 6. Optimize soil regions (preserve patch characteristics)
        soil_mask = self._optimize_soil_regions(soil_mask, image)
        
        if debug:
            return {
                'final_soil_mask': soil_mask,
                'debug_masks': debug_masks
            }
        
        return soil_mask
    
    def _detect_sky_morphological(self, image: np.ndarray) -> np.ndarray:
        """
        Detect sky based on morphology: large uniform bright regions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Detect bright regions (not too strict)
        _, bright_regions = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        
        # 2. Calculate local variance to detect uniformity
        # Sky regions should have small variance (uniform color)
        kernel_size = 15
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        
        # Calculate local mean
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        
        # Calculate local variance
        local_var = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
        
        # Low variance regions (uniform regions)
        uniform_mask = (local_var < 200).astype(np.uint8) * 255
        
        # 3. Combine brightness and uniformity
        sky_candidate = cv2.bitwise_and(bright_regions, uniform_mask)
        
        # 4. Morphological operations: remove small regions, keep large areas
        # Opening to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        sky_candidate = cv2.morphologyEx(sky_candidate, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        sky_candidate = cv2.morphologyEx(sky_candidate, cv2.MORPH_CLOSE, kernel)
        
        # 5. Region filtering: keep only large area regions
        contours, _ = cv2.findContours(sky_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sky_mask = np.zeros_like(gray)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Sky should be large area regions
            if area > (h * w) * 0.05:  # At least 5% of image area
                # Check region compactness (sky is usually continuous large blocks)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = (4 * np.pi * area) / (perimeter * perimeter)
                    if compactness > 0.1:  # Relatively regular shape
                        cv2.fillPoly(sky_mask, [contour], 255)
        
        # 6. Mainly look for sky in upper part
        sky_mask_filtered = np.zeros_like(sky_mask)
        sky_mask_filtered[:int(h*0.7), :] = sky_mask[:int(h*0.7), :]
        
        return sky_mask_filtered
    
    def _detect_vegetation_morphological(self, image: np.ndarray) -> np.ndarray:
        """
        Detect vegetation based on morphology: linear, striped textures
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Color pre-filtering (only as auxiliary, not main criterion)
        # Detect green tones, but with relatively loose range
        lower_green = np.array([35, 30, 20], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        green_hint = cv2.inRange(hsv, lower_green, upper_green)
        
        # 2. Linear texture detection
        vegetation_texture = self._detect_linear_texture(gray)
        
        # 3. Edge density detection (vegetation has many edges)
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate local edge density
        kernel_size = 10
        kernel = np.ones((kernel_size, kernel_size), np.float32)
        edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel) / (kernel_size * kernel_size * 255)
        
        # High edge density regions
        high_edge_density = (edge_density > 0.3).astype(np.uint8) * 255
        
        # 4. Gabor filter for striped texture detection
        gabor_response = self._apply_gabor_filters(gray)
        
        # 5. Combine all features
        # Vegetation = green hint AND (linear texture OR high edge density OR Gabor response)
        texture_features = cv2.bitwise_or(vegetation_texture, high_edge_density)
        texture_features = cv2.bitwise_or(texture_features, gabor_response)
        
        vegetation_mask = cv2.bitwise_and(green_hint, texture_features)
        
        # 6. Morphological optimization
        # Opening to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
        
        # Closing to connect nearby regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        
        return vegetation_mask
    
    def _detect_linear_texture(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect linear texture (grass stripe characteristics)
        """
        # Use linear structural elements in different directions
        linear_masks = []
        
        # Vertical linear structural element (detect vertical grass)
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        opening_v = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)
        linear_v = cv2.subtract(gray, opening_v)
        
        # Horizontal linear structural element
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        opening_h = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
        linear_h = cv2.subtract(gray, opening_h)
        
        # Diagonal linear structural element
        kernel_d1 = np.array([[1,0,0,0,0,0,1],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,1,0,0],
                              [0,0,0,1,0,0,0],
                              [0,0,1,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [1,0,0,0,0,0,1]], dtype=np.uint8)
        
        opening_d1 = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_d1)
        linear_d1 = cv2.subtract(gray, opening_d1)
        
        # Combine all directional linear features
        linear_texture = cv2.bitwise_or(linear_v, linear_h)
        linear_texture = cv2.bitwise_or(linear_texture, linear_d1)
        
        # Thresholding
        _, linear_texture = cv2.threshold(linear_texture, 20, 255, cv2.THRESH_BINARY)
        
        return linear_texture
    
    def _apply_gabor_filters(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Gabor filters to detect directional textures
        """
        gabor_responses = []
        
        # Gabor filters at different angles
        angles = [0, 30, 60, 90, 120, 150]
        
        for angle in angles:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel((15, 15), 3, np.radians(angle), 
                                       2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
            
            # Apply filter
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            gabor_responses.append(filtered)
        
        # Take maximum response across all directions
        gabor_max = np.maximum.reduce(gabor_responses)
        
        # Thresholding
        _, gabor_binary = cv2.threshold(gabor_max, 30, 255, cv2.THRESH_BINARY)
        
        return gabor_binary
    
    def _detect_tools_morphological(self, image: np.ndarray) -> np.ndarray:
        """
        Tool detection removed - avoid soil misclassification
        """
        # Return empty mask
        return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _optimize_soil_regions(self, soil_mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Optimize soil regions: preserve patch characteristics, remove small noise
        """
        # 1. Remove small isolated regions
        contours, _ = cv2.findContours(soil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(soil_mask)
        
        total_area = soil_mask.shape[0] * soil_mask.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Keep regions larger than 0.1% of total area
            if area > total_area * 0.001:
                cv2.fillPoly(filtered_mask, [contour], 255)
        
        # 2. Slight closing operation to connect nearby soil patches
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
        
        return filtered_mask
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray, 
                   mask_type: str = 'transparent') -> np.ndarray:
        """
        Apply mask to image
        
        Args:
            image: Input image
            mask: Mask image
            mask_type: Mask type ('transparent' or 'black')
            
        Returns:
            np.ndarray: Processed image
        """
        if mask_type == 'transparent':
            # Create RGBA image
            if image.shape[2] == 3:
                result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            else:
                result = image.copy()
            
            # Set transparency
            result[mask > 0, 3] = 0  # Completely transparent
            
        elif mask_type == 'black':
            result = image.copy()
            result[mask > 0] = [0, 0, 0]  # Set to black
            
        else:
            raise ValueError(f"Type de masque non supporté: {mask_type}")
        
        return result
    
    def process_image(self, image: np.ndarray, 
                     ruler_mask: Optional[np.ndarray] = None,
                     upper_boundary: Optional[int] = None,
                     mask_type: str = 'transparent') -> dict:
        """
        Complete image processing pipeline
        
        Args:
            image: Input image
            ruler_mask: Ruler region mask
            upper_boundary: Upper boundary y-coordinate (based on ruler 0 scale position)
            mask_type: Mask type
            
        Returns:
            dict: Processing results
        """
        # Segment soil areas
        soil_mask = self.segment_soil_area(image, ruler_mask, upper_boundary)
        
        # Create removal mask (non-soil areas)
        remove_mask = cv2.bitwise_not(soil_mask)
        
        # Apply mask
        processed_image = self.apply_mask(image, remove_mask, mask_type)
        
        return {
            'processed_image': processed_image,
            'soil_mask': soil_mask,
            'remove_mask': remove_mask,
            'detected_objects': []  # Initialize empty list for detected objects
        }
    
    def visualize_debug(self, debug_result: dict, save_path: str = None):
        """
        Visualize debug results
        """
        import matplotlib.pyplot as plt
        
        masks = debug_result['debug_masks']
        final_mask = debug_result['final_soil_mask']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        # Show each detection step
        titles = ['Limite Supérieure', 'Ciel', 'Végétation', 'Sol Final', 'Zones Supprimées', 'Réservé']
        mask_keys = ['upper_boundary', 'sky', 'vegetation']
        
        for i, (key, title) in enumerate(zip(mask_keys, titles)):
            if key in masks:
                axes[i].imshow(masks[key], cmap='gray')
                axes[i].set_title(title)
                axes[i].axis('off')
        
        # Show final result
        axes[3].imshow(final_mask, cmap='gray')
        axes[3].set_title('Sol Final')
        axes[3].axis('off')
        
        axes[4].imshow(255 - final_mask, cmap='gray')
        axes[4].set_title('Zones Supprimées')
        axes[4].axis('off')
        
        # Hide extra subplots
        axes[5].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print statistics
        total_pixels = final_mask.shape[0] * final_mask.shape[1]
        soil_pixels = np.sum(final_mask > 0)
        soil_percentage = (soil_pixels / total_pixels) * 100
        
        print(f"\n=== Statistiques de Segmentation ===")
        print(f"Pixels totaux: {total_pixels}")
        print(f"Pixels de sol: {soil_pixels}")
        print(f"Pourcentage de sol: {soil_percentage:.2f}%")
        
        for key, mask in masks.items():
            removed_pixels = np.sum(mask > 0)
            removed_percentage = (removed_pixels / total_pixels) * 100
            print(f"{key} supprimé: {removed_pixels} ({removed_percentage:.2f}%)")


# For compatibility, create an alias
SoilSegmentation = MorphologicalSoilSegmentation


# Usage example
def test_morphological_segmentation():
    """Test function"""
    
    # Create segmenter
    segmenter = MorphologicalSoilSegmentation()
    
    # Load image
    image = cv2.imread('your_image.jpg')
    
    # Run debug mode
    debug_result = segmenter.segment_soil_area(image, debug=True)
    
    # Visualize results
    segmenter.visualize_debug(debug_result)
    
    return debug_result

# To use, call:
# result = test_morphological_segmentation()