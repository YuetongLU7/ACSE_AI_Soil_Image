import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import shutil
from .ruler_detection import RulerDetector
from .soil_segmentation import SoilSegmentation
from .quality_assessment import ImageQualityAssessment

class SoilImagePreprocessor:
    """Soil Image Preprocessing Pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Preprocessing Pipeline
        
        Args:
            config_path: Configuration file path
        """
        self.config = self._load_config(config_path)
        self.ruler_detector = RulerDetector(
            min_length=self.config['preprocessing']['ruler_detection']['min_length'],
            max_length=self.config['preprocessing']['ruler_detection']['max_length']
        )
        self.soil_segmentation = SoilSegmentation()
        self.quality_assessor = ImageQualityAssessment(self.config)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load the configuration file"""
        with open(config_path, 'r', encoding='utf-8') as f:   # r means read mode
            return yaml.safe_load(f)
    
    def process_single_image(self, image_path: str, 
                           output_dir: str = None) -> Dict:
        """
        Process a single image
        
        Args:
            image_path: Input image path
            output_dir: Output directory
            
        Returns:
            Dict: Processing result
        """
        # Read the image with proper encoding support for Chinese characters
        try:
            # Use np.fromfile for better Unicode support
            image_data = np.fromfile(image_path, dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot read the image: {image_path}")
        except Exception as e:
            raise ValueError(f"Cannot read the image: {image_path}. Error: {str(e)}")
        
        image_name = Path(image_path).stem   # stem means the file name without the extension
        
        # Step 1: Detect the ruler
        ruler_info = self.ruler_detector.detect_ruler(image)
        if ruler_info is None:
            print(f"Warning: No ruler detected in the image {image_name}")
            ruler_mask = None
        else:
            ruler_mask = self.ruler_detector.extract_ruler_region(image, ruler_info)
            print(f"Successfully detected ruler，scale ratio: {ruler_info['scale_ratio']:.2f} pixel/cm")
        
        # 步骤2: 土壤区域分割和物体移除
        segmentation_result = self.soil_segmentation.process_image(
            image, 
            ruler_mask, 
            mask_type=self.config['preprocessing']['soil_segmentation']['mask_type']
        )
        
        # 收集结果
        result = {
            'image_name': image_name,
            'original_image': image,
            'ruler_info': ruler_info,
            'processed_image': segmentation_result['processed_image'],
            'soil_mask': segmentation_result['soil_mask'],
            'remove_mask': segmentation_result['remove_mask'],
            'detected_objects': segmentation_result['detected_objects']
        }
        
        # 保存结果
        if output_dir:
            self._save_results(result, output_dir)
        
        return result
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     clear_output: bool = True, 
                     incremental: bool = False) -> List[Dict]:
        """
        批量处理图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            clear_output: 是否清空输出目录
            incremental: 是否只处理新图像
            
        Returns:
            List[Dict]: 处理结果列表
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 清空输出目录
        if clear_output and output_path.exists():
            print(f"清空输出目录: {output_path}")
            shutil.rmtree(output_path)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 找到所有图像文件
        all_image_files = [f for f in input_path.iterdir() 
                          if f.suffix.lower() in image_extensions]
        
        # 增量处理：只处理未处理过的图像
        if incremental and not clear_output:
            processed_names = set()
            for f in output_path.glob("*_processed.png"):
                # 从 "filename_processed.png" 提取原始文件名
                original_name = f.name.replace("_processed.png", "")
                processed_names.add(original_name)
            
            image_files = [f for f in all_image_files 
                          if f.stem not in processed_names]
            
            if image_files:
                print(f"增量处理模式: 跳过 {len(all_image_files) - len(image_files)} 个已处理文件")
            else:
                print("增量处理模式: 所有文件已处理完成")
                return []
        else:
            image_files = all_image_files
        
        if not image_files:
            print(f"在目录 {input_dir} 中未找到图像文件")
            return []
        
        results = []
        successful_count = 0
        failed_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            try:
                print(f"处理图像 ({i}/{len(image_files)}): {image_file.name}")
                result = self.process_single_image(str(image_file), str(output_path))
                results.append(result)
                successful_count += 1
                
                # 显示检测结果
                if result['ruler_info'] and result['ruler_info']['ruler_detected']:
                    method = result['ruler_info'].get('detection_method', 'unknown')
                    confidence = result['ruler_info'].get('confidence', 0)
                    print(f"  ✓ 米尺检测成功 - 方法: {method}, 置信度: {confidence:.2f}, 比例: {result['ruler_info']['scale_ratio']:.2f} px/cm")
                else:
                    print(f"  ✗ 未检测到米尺")
                    
                print(f"  检测到 {len(result['detected_objects'])} 个物体")
                
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                failed_count += 1
                
                # 创建失败记录
                failed_result = {
                    'image_name': image_file.stem,
                    'status': 'failed',
                    'error': str(e),
                    'ruler_info': None,
                    'detected_objects': []
                }
                results.append(failed_result)
                continue
        
        print(f"\n批处理完成: 成功 {successful_count} 张, 失败 {failed_count} 张")
        
        # 质量过滤
        if self.quality_assessor.enable_filtering:
            print("开始质量评估和过滤...")
            high_quality_results, low_quality_results = self.quality_assessor.filter_low_quality_images(results)
            
            print(f"质量过滤结果:")
            print(f"  高质量图像: {len(high_quality_results)} 张")
            print(f"  低质量图像: {len(low_quality_results)} 张")
            
            # 移动低质量图像到单独文件夹
            if low_quality_results:
                self._handle_low_quality_images(low_quality_results, output_path)
            
            # 保存质量过滤报告
            self._save_quality_report(high_quality_results, low_quality_results, output_path)
            
            # 只返回高质量结果用于后续处理
            filtered_results = high_quality_results
        else:
            filtered_results = results
        
        # 保存批处理报告
        self._save_batch_report(filtered_results, output_path)
        
        return filtered_results
    
    def _serialize_ruler_info(self, ruler_info: Dict) -> Dict:
        """将米尺信息序列化为JSON可保存格式"""
        if ruler_info is None:
            return None
        
        serialized = {}
        for key, value in ruler_info.items():
            if isinstance(value, np.ndarray):
                # 转换numpy数组为列表
                serialized[key] = value.tolist()
            else:
                serialized[key] = value
        return serialized
    
    def _save_results(self, result: Dict, output_dir: str):
        """保存处理结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_name = result['image_name']
        
        # 保存处理后的图像
        processed_image_path = output_path / f"{image_name}_processed.png"
        cv2.imwrite(str(processed_image_path), result['processed_image'])
        
        # 保存土壤掩码
        soil_mask_path = output_path / f"{image_name}_soil_mask.png"
        cv2.imwrite(str(soil_mask_path), result['soil_mask'])
        
        # 保存移除掩码
        remove_mask_path = output_path / f"{image_name}_remove_mask.png"
        cv2.imwrite(str(remove_mask_path), result['remove_mask'])
        
        # 保存可视化结果（米尺检测可视化）
        if result['ruler_info'] and result['ruler_info']['ruler_detected']:
            # 使用原始图像进行可视化，确保不影响处理后的图像
            viz_image = self.ruler_detector.visualize_detection(
                result['original_image'].copy(), result['ruler_info']
            )
            viz_path = output_path / f"{image_name}_ruler_detection.png"
            cv2.imwrite(str(viz_path), viz_image)
        
        # 保存元数据
        metadata = {
            'image_name': image_name,
            'ruler_info': self._serialize_ruler_info(result['ruler_info']),
            'detected_objects': result['detected_objects'],
            'processing_config': self.config
        }
        
        metadata_path = output_path / f"{image_name}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def _save_batch_report(self, results: List[Dict], output_dir: Path):
        """保存批处理报告"""
        successful_results = [r for r in results if r.get('status') != 'failed']
        failed_results = [r for r in results if r.get('status') == 'failed']
        
        successful_ruler_detections = [r for r in successful_results 
                                     if r.get('ruler_info') and r['ruler_info']['ruler_detected']]
        
        # 计算平均比例（避免空列表）
        if successful_ruler_detections:
            avg_scale_ratio = np.mean([r['ruler_info']['scale_ratio'] 
                                     for r in successful_ruler_detections])
        else:
            avg_scale_ratio = 0.0
        
        report = {
            'total_images': len(results),
            'successful_count': len(successful_results),
            'failed_count': len(failed_results),
            'successful_ruler_detections': len(successful_ruler_detections),
            'ruler_detection_rate': len(successful_ruler_detections) / len(results) * 100 if results else 0,
            'average_scale_ratio': float(avg_scale_ratio),
            'detection_methods_used': self._summarize_detection_methods(successful_ruler_detections),
            'detected_objects_summary': self._summarize_detected_objects(successful_results),
            'failed_images': [{'name': r['image_name'], 'error': r['error']} for r in failed_results]
        }
        
        report_path = output_dir / "batch_processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def _summarize_detected_objects(self, results: List[Dict]) -> Dict:
        """汇总检测到的物体统计信息"""
        object_counts = {}
        for result in results:
            for obj in result['detected_objects']:
                class_name = obj['class_name']
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1
        
        return object_counts
    
    def _summarize_detection_methods(self, successful_detections: List[Dict]) -> Dict:
        """统计检测方法使用情况"""
        method_counts = {}
        for result in successful_detections:
            method = result['ruler_info'].get('detection_method', 'unknown')
            if method not in method_counts:
                method_counts[method] = 0
            method_counts[method] += 1
        
        return method_counts
    
    def _handle_low_quality_images(self, low_quality_results: List[Dict], output_path: Path):
        """处理低质量图像 - 移动到单独文件夹"""
        low_quality_dir = output_path / "low_quality"
        low_quality_dir.mkdir(exist_ok=True)
        
        for result in low_quality_results:
            if result.get('status') == 'failed':
                continue
                
            image_name = result['image_name']
            
            # 移动相关文件到低质量文件夹
            file_patterns = [
                f"{image_name}_processed.png",
                f"{image_name}_soil_mask.png", 
                f"{image_name}_remove_mask.png",
                f"{image_name}_ruler_detection.png",
                f"{image_name}_metadata.json"
            ]
            
            for pattern in file_patterns:
                source_file = output_path / pattern
                if source_file.exists():
                    target_file = low_quality_dir / pattern
                    source_file.rename(target_file)
    
    def _save_quality_report(self, high_quality_results: List[Dict], 
                           low_quality_results: List[Dict], output_path: Path):
        """保存质量评估报告"""
        
        # 统计信息
        total_count = len(high_quality_results) + len(low_quality_results)
        high_quality_count = len(high_quality_results)
        low_quality_count = len(low_quality_results)
        
        # 收集低质量图像的问题统计
        problem_stats = {}
        quality_scores = []
        
        for result in low_quality_results:
            if result.get('status') == 'failed':
                continue
                
            assessment = result.get('quality_assessment', {})
            issues = assessment.get('issues', [])
            quality_score = assessment.get('quality_score', 0)
            
            quality_scores.append(quality_score)
            
            for issue in issues:
                # 提取问题类型
                if "土壤覆盖率" in issue:
                    problem_type = "土壤覆盖率过低"
                elif "反光区域" in issue:
                    problem_type = "反光区域过多"
                elif "对比度" in issue:
                    problem_type = "对比度不足"
                elif "阴影区域" in issue:
                    problem_type = "阴影区域过多"
                elif "连通性" in issue:
                    problem_type = "掩码连通性差"
                else:
                    problem_type = "其他问题"
                
                if problem_type not in problem_stats:
                    problem_stats[problem_type] = 0
                problem_stats[problem_type] += 1
        
        # 创建报告
        report = {
            'summary': {
                'total_images': total_count,
                'high_quality_count': high_quality_count,
                'low_quality_count': low_quality_count,
                'quality_pass_rate': high_quality_count / total_count * 100 if total_count > 0 else 0
            },
            'quality_thresholds': {
                'min_soil_coverage': self.quality_assessor.min_soil_coverage,
                'max_reflection_ratio': self.quality_assessor.max_reflection_ratio,
                'min_contrast': self.quality_assessor.min_contrast,
                'max_shadow_ratio': self.quality_assessor.max_shadow_ratio,
                'min_mask_connectivity': self.quality_assessor.min_mask_connectivity
            },
            'problem_statistics': problem_stats,
            'low_quality_details': []
        }
        
        # 添加低质量图像详情
        for result in low_quality_results:
            if result.get('status') == 'failed':
                continue
                
            assessment = result.get('quality_assessment', {})
            
            detail = {
                'image_name': result['image_name'],
                'quality_score': assessment.get('quality_score', 0),
                'issues': assessment.get('issues', []),
                'metrics': assessment.get('metrics', {})
            }
            
            report['low_quality_details'].append(detail)
        
        # 保存报告
        report_path = output_path / "quality_assessment_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # 打印摘要
        print(f"\\n质量评估摘要:")
        print(f"  总图像数: {total_count}")
        print(f"  高质量: {high_quality_count} ({high_quality_count/total_count*100:.1f}%)")
        print(f"  低质量: {low_quality_count} ({low_quality_count/total_count*100:.1f}%)")
        
        if problem_stats:
            print(f"\\n主要质量问题:")
            for problem, count in sorted(problem_stats.items(), key=lambda x: x[1], reverse=True):
                print(f"  {problem}: {count} 张图像")
    
    def create_training_dataset(self, processed_results: List[Dict], 
                              output_dir: str, 
                              train_ratio: float = 0.8):
        """
        创建用于AI训练的数据集
        
        Args:
            processed_results: 预处理结果列表
            output_dir: 输出目录
            train_ratio: 训练集比例
        """
        output_path = Path(output_dir)
        train_dir = output_path / "train"
        val_dir = output_path / "val"
        
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # 随机分割训练集和验证集
        np.random.shuffle(processed_results)
        split_idx = int(len(processed_results) * train_ratio)
        
        train_data = processed_results[:split_idx]
        val_data = processed_results[split_idx:]
        
        # 创建训练集
        self._create_dataset_split(train_data, train_dir)
        
        # 创建验证集
        self._create_dataset_split(val_data, val_dir)
        
        print(f"数据集创建完成: 训练集 {len(train_data)} 张, 验证集 {len(val_data)} 张")
    
    def _create_dataset_split(self, data: List[Dict], output_dir: Path):
        """创建数据集分割"""
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        for result in data:
            image_name = result['image_name']
            
            # 保存处理后的图像
            image_path = images_dir / f"{image_name}.png"
            cv2.imwrite(str(image_path), result['processed_image'])
            
            # 保存土壤掩码
            mask_path = masks_dir / f"{image_name}.png"
            cv2.imwrite(str(mask_path), result['soil_mask'])