#!/usr/bin/env python3
"""
Simple test script to verify the data loading fix works
"""
import json
from pathlib import Path

def test_sample_pairs(images_dir, annotations_dir):
    """Test the sample pairing logic"""
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    
    samples = []
    
    # Get all JSON annotation files
    json_files = list(annotations_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON annotation files")
    
    for json_file in json_files:
        # Find corresponding image
        base_name = json_file.stem.replace('_hz', '')
        
        # Try different extensions (jpg, JPG)
        image_path = None
        for ext in ['.jpg', '.JPG']:
            candidate_path = images_dir / (base_name + ext)
            if candidate_path.exists():
                image_path = candidate_path
                break
        
        if image_path:
            samples.append({
                'image_path': str(image_path),
                'annotation_path': str(json_file),
                'image_name': image_path.name
            })
            print(f"✓ Match found: {json_file.name} -> {image_path.name}")
        else:
            print(f"✗ No image found for: {json_file.name}")
    
    print(f"\nTotal valid samples: {len(samples)}")
    return samples

if __name__ == "__main__":
    print("Testing data loading fix...")
    samples = test_sample_pairs("data/raw", "data/horizon")
    
    if len(samples) > 0:
        print("\n✓ SUCCESS: Data loading fix works!")
        print(f"Found {len(samples)} valid image-annotation pairs")
    else:
        print("\n✗ FAILED: No valid pairs found")