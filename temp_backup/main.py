#!/usr/bin/env python3
"""
土壤图像预处理主程序
"""

import argparse
import sys
from pathlib import Path
from src.preprocessing import SoilImagePreprocessor

def main():
    parser = argparse.ArgumentParser(description='土壤图像预处理工具')
    parser.add_argument('--input', '-i', required=True, 
                       help='输入图像或目录路径')
    parser.add_argument('--output', '-o', required=True,
                       help='输出目录路径')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='批处理模式')
    parser.add_argument('--create-dataset', action='store_true',
                       help='创建训练数据集')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例（用于数据集创建）')
    parser.add_argument('--no-clear', action='store_true',
                       help='不清空输出目录')
    parser.add_argument('--incremental', action='store_true',
                       help='增量处理模式（只处理新图像）')
    parser.add_argument('--disable-quality-filter', action='store_true',
                       help='禁用质量过滤')
    
    args = parser.parse_args()
    
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {args.input}")
        sys.exit(1)
    
    # 创建预处理器
    try:
        preprocessor = SoilImagePreprocessor(args.config)
        
        # 根据命令行参数调整质量过滤设置
        if args.disable_quality_filter:
            preprocessor.quality_assessor.enable_filtering = False
            print("质量过滤已禁用")
        
        print(f"使用配置文件: {args.config}")
    except Exception as e:
        print(f"错误: 无法加载配置文件: {e}")
        sys.exit(1)
    
    # 处理图像
    if args.batch or input_path.is_dir():
        clear_output = not args.no_clear
        if clear_output:
            print("开始批处理（清空输出目录）...")
        elif args.incremental:
            print("开始增量批处理...")
        else:
            print("开始批处理（保留现有文件）...")
            
        results = preprocessor.process_batch(
            str(input_path), 
            args.output, 
            clear_output=clear_output,
            incremental=args.incremental
        )
        print(f"批处理完成，共处理 {len(results)} 张图像")
        
        # 创建训练数据集
        if args.create_dataset:
            dataset_dir = Path(args.output) / "dataset"
            preprocessor.create_training_dataset(results, str(dataset_dir), args.train_ratio)
            
    else:
        print("开始处理单张图像...")
        result = preprocessor.process_single_image(str(input_path), args.output)
        print(f"图像处理完成: {result['image_name']}")
        
        if result['ruler_info'] and result['ruler_info']['ruler_detected']:
            print(f"米尺检测成功，比例: {result['ruler_info']['scale_ratio']:.2f} 像素/厘米")
        else:
            print("未检测到米尺")
            
        print(f"检测到 {len(result['detected_objects'])} 个物体")

if __name__ == "__main__":
    main()