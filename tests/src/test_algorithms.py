#!/usr/bin/env python3
"""
测试改进后的米尺检测和土壤分割算法
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ruler_detection import RulerDetector
from src.soil_segmentation import SoilSegmentation

class ImprovedAlgorithmTester:
    def __init__(self):
        self.ruler_detector = RulerDetector()
        self.soil_segmenter = SoilSegmentation()
        
    def test_single_image(self, image_path: str, save_results: bool = True):
        """测试单张图像的处理效果"""
        print(f"处理图像: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
            
        # 1. 米尺检测
        print("1. 开始米尺检测...")
        ruler_info = self.ruler_detector.detect_ruler(image)
        
        if ruler_info and ruler_info['ruler_detected']:
            print(f"   ✓ 米尺检测成功")
            print(f"   - 检测方法: {ruler_info['detection_method']}")
            print(f"   - 置信度: {ruler_info['confidence']:.2f}")
            print(f"   - 像素长度: {ruler_info['pixel_length']:.1f}")
            print(f"   - 比例: {ruler_info['scale_ratio']:.2f} 像素/厘米")
            if 'aspect_ratio' in ruler_info:
                print(f"   - 长宽比: {ruler_info['aspect_ratio']:.1f}")
            if 'angle' in ruler_info:
                print(f"   - 角度: {ruler_info['angle']:.1f}°")
        else:
            print("   ✗ 米尺检测失败")
            
        # 2. 土壤区域分割
        print("2. 开始土壤区域分割...")
        soil_result = self.soil_segmenter.process_image(image)
        
        print(f"   ✓ 土壤分割完成")
        print(f"   - 检测到的物体数量: {len(soil_result['detected_objects'])}")
        
        # 计算土壤区域比例
        soil_area = np.sum(soil_result['soil_mask'] > 0)
        total_area = soil_result['soil_mask'].shape[0] * soil_result['soil_mask'].shape[1]
        soil_ratio = soil_area / total_area
        print(f"   - 土壤区域比例: {soil_ratio:.2%}")
        
        # 3. 可视化结果
        if save_results:
            self.save_visualization(image, ruler_info, soil_result, image_path)
            
        return ruler_info, soil_result
    
    def save_visualization(self, image, ruler_info, soil_result, image_path):
        """保存可视化结果"""
        base_name = Path(image_path).stem
        output_dir = Path("tests/data/results")
        output_dir.mkdir(exist_ok=True)
        
        # 创建4个子图的可视化
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 原图
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("原始图像")
        axes[0, 0].axis('off')
        
        # 米尺检测结果
        if ruler_info and ruler_info['ruler_detected']:
            ruler_vis = self.ruler_detector.visualize_detection(image, ruler_info)
            axes[0, 1].imshow(cv2.cvtColor(ruler_vis, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f"米尺检测 (置信度: {ruler_info['confidence']:.2f})")
        else:
            axes[0, 1].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title("米尺检测失败")
        axes[0, 1].axis('off')
        
        # 土壤掩码
        axes[1, 0].imshow(soil_result['soil_mask'], cmap='gray')
        axes[1, 0].set_title("土壤区域掩码")
        axes[1, 0].axis('off')
        
        # 处理后的图像
        processed_rgb = cv2.cvtColor(soil_result['processed_image'], cv2.COLOR_BGR2RGB)
        axes[1, 1].imshow(processed_rgb)
        axes[1, 1].set_title("处理后图像")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f"{base_name}_results.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✓ 可视化结果已保存: {output_path}")
        
        # 单独保存掩码
        mask_path = output_dir / f"{base_name}_soil_mask.png"
        cv2.imwrite(str(mask_path), soil_result['soil_mask'])
        
        # 单独保存处理后的图像
        processed_path = output_dir / f"{base_name}_processed.png"
        cv2.imwrite(str(processed_path), soil_result['processed_image'])
    
    def test_batch_images(self, image_dir: str):
        """批量测试图像"""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            print(f"图像目录不存在: {image_dir}")
            return
            
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(image_dir.glob(f"*{ext}")))
            image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
            
        if not image_files:
            print(f"在 {image_dir} 中未找到图像文件")
            return
            
        print(f"找到 {len(image_files)} 张图像")
        
        # 统计结果
        ruler_success = 0
        total_images = len(image_files)
        
        for i, image_file in enumerate(image_files):
            print(f"\n{'='*50}")
            print(f"进度: {i+1}/{total_images}")
            
            ruler_info, soil_result = self.test_single_image(str(image_file))
            
            if ruler_info and ruler_info['ruler_detected']:
                ruler_success += 1
                
        print(f"\n{'='*50}")
        print("批量测试完成")
        print(f"米尺检测成功率: {ruler_success}/{total_images} ({ruler_success/total_images:.1%})")

def main():
    """主测试函数"""
    tester = ImprovedAlgorithmTester()
    
    # 测试单张图像
    print("=== 改进算法测试 ===")
    
    # 检查是否有测试图像
    test_image_dirs = [
        "data/raw",
        "data/processed", 
        "tests/data/images"
    ]
    
    found_images = False
    for dir_path in test_image_dirs:
        if os.path.exists(dir_path):
            # 寻找第一张图像进行测试
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                images = list(Path(dir_path).glob(f"*{ext}"))
                if images:
                    print(f"使用测试图像: {images[0]}")
                    tester.test_single_image(str(images[0]))
                    found_images = True
                    break
            if found_images:
                break
    
    if not found_images:
        print("未找到测试图像")
        print("请在以下目录之一放置图像文件:")
        for dir_path in test_image_dirs:
            print(f"  - {dir_path}")
        return
    
    # 询问是否进行批量测试
    print("\n是否进行批量测试? (y/n)")
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        for dir_path in test_image_dirs:
            if os.path.exists(dir_path):
                print(f"\n批量测试目录: {dir_path}")
                tester.test_batch_images(dir_path)
                break

if __name__ == "__main__":
    main()