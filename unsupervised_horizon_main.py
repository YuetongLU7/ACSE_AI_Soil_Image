#!/usr/bin/env python3
"""
无监督土壤层位检测主程序
基于纹理和颜色变化检测土壤层位边界，无需标签数据
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import json
from typing import Optional

from src.unsupervised_horizon_detection import UnsupervisedHorizonDetector


def load_soil_mask(image_path: str) -> Optional[np.ndarray]:
    """加载土壤掩码"""
    image_path = Path(image_path)
    mask_path = image_path.parent / f"{image_path.stem.replace('_processed', '')}_soil_mask.png"
    
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return (mask > 128).astype(np.uint8)
    else:
        print(f"警告: 未找到土壤掩码 {mask_path}, 使用全图")
        return None


def load_original_image(processed_path: str) -> Optional[np.ndarray]:
    """加载原始图像用于可视化"""
    processed_path = Path(processed_path)
    
    # 尝试找到原始图像
    original_candidates = [
        processed_path.parent.parent / "raw" / f"{processed_path.stem.replace('_processed', '')}.jpg",
        processed_path.parent.parent / "raw" / f"{processed_path.stem.replace('_processed', '')}.png",
        processed_path.parent.parent / "raw" / f"{processed_path.stem.replace('_processed', '')}.jpeg",
        # 如果原始图像在同一目录
        processed_path.parent / f"{processed_path.stem.replace('_processed', '')}.jpg",
        processed_path.parent / f"{processed_path.stem.replace('_processed', '')}.png",
    ]
    
    for original_path in original_candidates:
        if original_path.exists():
            original = cv2.imread(str(original_path))
            if original is not None:
                print(f"找到原始图像: {original_path}")
                return cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    print("警告: 未找到原始图像，使用处理后的图像")
    return None


def load_metadata(image_path: str) -> Optional[dict]:
    """加载图像元数据"""
    image_path = Path(image_path)
    metadata_path = image_path.parent / f"{image_path.stem.replace('_processed', '')}_metadata.json"
    
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                print(f"加载元数据: {metadata_path}")
                return metadata
        except Exception as e:
            print(f"警告: 无法加载元数据 {metadata_path}: {e}")
    else:
        print(f"警告: 未找到元数据 {metadata_path}")
    
    return None


def process_single_image(image_path: str, output_dir: str, config: Optional[dict] = None):
    """处理单张图像"""
    print(f"处理图像: {image_path}")
    
    # 检查文件是否存在  
    if not Path(image_path).exists():
        print(f"错误: 图像文件不存在: {image_path}")
        return None
    
    # 加载处理后的图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法加载图像: {image_path}")
        return None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"处理后图像尺寸: {image.shape}")
    
    # 加载原始图像用于可视化
    original_image = load_original_image(image_path)
    
    # 加载土壤掩码
    mask = load_soil_mask(image_path)
    if mask is not None:
        print(f"加载土壤掩码，尺寸: {mask.shape}")
    
    # 加载元数据
    metadata = load_metadata(image_path)
    
    # 创建检测器
    detector = UnsupervisedHorizonDetector(config)
    
    # 检测层位
    print("开始无监督层位检测...")
    results = detector.detect_horizons(image, mask, metadata, original_image)
    
    # 显示结果
    print(f"\n=== 检测结果 ===")
    print(f"检测到 {results['num_horizons']} 个层位边界")
    
    if 'soil_region' in results:
        soil_region = results['soil_region']
        print(f"土壤有效区域: y={soil_region['top']} 到 y={soil_region['bottom']}")
    
    if results['horizons']:
        print("\n层位详情:")
        for i, horizon in enumerate(results['horizons']):
            print(f"  层位 {i+1}: ")
            print(f"    边界位置: y = {horizon['boundary_y']} 像素")
            print(f"    置信度: {horizon['confidence']:.3f}")
            print(f"    信号强度: {horizon.get('signal_strength', 0):.3f}")
            
            # 如果有尺子信息，显示深度
            if metadata and 'ruler_info' in metadata and metadata['ruler_info']['ruler_detected']:
                ruler_info = metadata['ruler_info']
                if 'detected_digits' in ruler_info and ruler_info['detected_digits']:
                    min_digit = min(ruler_info['detected_digits'], key=lambda x: x['value'])
                    zero_y = min_digit['center_y'] - (min_digit['value'] * ruler_info.get('scale_ratio', 15))
                    depth_cm = max(0, (horizon['boundary_y'] - zero_y) / ruler_info.get('scale_ratio', 15))
                    print(f"    估计深度: {depth_cm:.1f} 厘米")
    else:
        print("未检测到明显的层位边界")
    
    # 保存结果
    vis_path, data_path = detector.save_results(results, image_path, output_dir)
    
    return {
        'image_path': image_path,
        'results': results,
        'visualization_path': vis_path,
        'data_path': data_path
    }


def process_batch(input_dir: str, output_dir: str, config: Optional[dict] = None):
    """批量处理图像"""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return []
    
    # 查找处理过的图像
    processed_images = list(input_path.glob("*_processed.png"))
    if not processed_images:
        # 如果没有找到处理过的图像，查找原始图像
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        processed_images = []
        for ext in image_extensions:
            processed_images.extend(input_path.glob(ext))
    
    if not processed_images:
        print(f"错误: 在目录 {input_dir} 中未找到图像文件")
        return []
    
    print(f"找到 {len(processed_images)} 张图像待处理")
    
    results = []
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(processed_images, 1):
        print(f"\n处理进度: {i}/{len(processed_images)}")
        try:
            result = process_single_image(str(image_path), output_dir, config)
            if result:
                results.append(result)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"处理 {image_path} 时出错: {e}")
            failed += 1
    
    print(f"\n=== 批处理完成 ===")
    print(f"成功处理: {successful} 张")
    print(f"处理失败: {failed} 张")
    print(f"结果保存在: {output_dir}")
    
    return results


def create_custom_config():
    """创建自定义配置"""
    config = {
        'min_horizon_height': 40,      # 进一步增加最小层位高度
        'max_horizons': 5,             # 减少最大层位数量，更符合实际
        'smoothing_window': 31,        # 增加平滑窗口
        'peak_prominence': 0.25,       # 增加峰值显著性阈值
        'clustering_method': 'kmeans',
        'n_clusters': 4,               # 减少聚类数量
        'feature_weights': {
            'texture': 0.3,            # 调整权重
            'color': 0.4,              # 颜色权重更高
            'gradient': 0.3
        }
    }
    return config


def main():
    parser = argparse.ArgumentParser(description='无监督土壤层位检测工具')
    parser.add_argument('--input', '-i', required=True,
                       help='输入图像文件或目录路径')
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录路径')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='批处理模式')
    parser.add_argument('--config', '-c', type=str,
                       help='自定义配置文件路径(JSON格式)')
    parser.add_argument('--custom-config', action='store_true',
                       help='使用优化的自定义配置')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config = None
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"加载配置文件: {args.config}")
        except Exception as e:
            print(f"错误: 无法加载配置文件: {e}")
            sys.exit(1)
    elif args.custom_config:
        config = create_custom_config()
        print("使用优化的自定义配置")
    
    # 处理图像
    input_path = Path(args.input)
    
    if args.batch or input_path.is_dir():
        # 批处理模式
        results = process_batch(str(input_path), str(output_dir), config)
        
        # 保存批处理摘要
        if results:
            summary = {
                'total_processed': len(results),
                'average_horizons': np.mean([r['results']['num_horizons'] for r in results]),
                'images': [
                    {
                        'image': Path(r['image_path']).name,
                        'num_horizons': r['results']['num_horizons'],
                        'confidence_scores': r['results']['confidence_scores']
                    }
                    for r in results
                ]
            }
            
            summary_path = output_dir / "batch_processing_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"\n批处理摘要已保存: {summary_path}")
            print(f"平均检测层位数: {summary['average_horizons']:.1f}")
    
    else:
        # 单图像处理模式
        if not input_path.exists():
            print(f"错误: 输入文件不存在: {args.input}")
            sys.exit(1)
        
        result = process_single_image(str(input_path), str(output_dir), config)
        if result:
            print(f"\n处理完成! 结果保存在:")
            print(f"  可视化: {result['visualization_path']}")
            print(f"  数据: {result['data_path']}")
        else:
            print("处理失败")
            sys.exit(1)


if __name__ == "__main__":
    main()