import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from skimage.measure import label, regionprops
import logging

class ImageQualityAssessment:
    """图像质量评估器 - 评估土壤图像和掩码的质量"""
    
    def __init__(self, config: Dict):
        """
        初始化质量评估器
        
        Args:
            config: 质量控制配置参数
        """
        self.config = config.get('quality_control', {})
        self.enable_filtering = self.config.get('enable_filtering', True)
        
        # 质量阈值（为土壤剖面图像优化）
        self.min_soil_coverage = self.config.get('min_soil_coverage', 0.05)  # 降低到5%
        self.max_reflection_ratio = self.config.get('max_reflection_ratio', 0.4)  # 允许更多反光
        self.min_contrast = self.config.get('min_contrast', 15)  # 降低对比度要求
        self.max_shadow_ratio = self.config.get('max_shadow_ratio', 0.6)  # 允许更多阴影
        self.min_mask_connectivity = self.config.get('min_mask_connectivity', 0.3)  # 降低连通性要求
        
    def assess_image_quality(self, image: np.ndarray, soil_mask: np.ndarray) -> Dict:
        """
        评估图像和土壤掩码的整体质量
        
        Args:
            image: 原始图像
            soil_mask: 土壤掩码
            
        Returns:
            Dict: 质量评估结果
        """
        if not self.enable_filtering:
            return {
                'is_good_quality': True,
                'quality_score': 1.0,
                'issues': [],
                'metrics': {}
            }
        
        # 计算各项质量指标
        soil_coverage = self._calculate_soil_coverage(soil_mask)
        reflection_ratio = self._detect_reflection_areas(image)
        contrast_score = self._calculate_contrast(image)
        shadow_ratio = self._detect_shadow_areas(image)
        connectivity_score = self._calculate_mask_connectivity(soil_mask)
        
        # 收集指标
        metrics = {
            'soil_coverage': soil_coverage,
            'reflection_ratio': reflection_ratio,
            'contrast_score': contrast_score,
            'shadow_ratio': shadow_ratio,
            'connectivity_score': connectivity_score
        }
        
        # 检查质量问题
        issues = []
        
        if soil_coverage < self.min_soil_coverage:
            issues.append(f"土壤覆盖率过低: {soil_coverage:.2%} < {self.min_soil_coverage:.2%}")
        
        if reflection_ratio > self.max_reflection_ratio:
            issues.append(f"反光区域过多: {reflection_ratio:.2%} > {self.max_reflection_ratio:.2%}")
        
        if contrast_score < self.min_contrast:
            issues.append(f"对比度不足: {contrast_score:.1f} < {self.min_contrast}")
        
        if shadow_ratio > self.max_shadow_ratio:
            issues.append(f"阴影区域过多: {shadow_ratio:.2%} > {self.max_shadow_ratio:.2%}")
        
        if connectivity_score < self.min_mask_connectivity:
            issues.append(f"掩码连通性差: {connectivity_score:.2%} < {self.min_mask_connectivity:.2%}")
        
        # 计算综合质量分数
        quality_score = self._calculate_overall_quality_score(metrics)
        
        # 判断是否为良好质量
        is_good_quality = len(issues) == 0
        
        return {
            'is_good_quality': is_good_quality,
            'quality_score': quality_score,
            'issues': issues,
            'metrics': metrics
        }
    
    def _calculate_soil_coverage(self, soil_mask: np.ndarray) -> float:
        """计算土壤覆盖率"""
        if soil_mask.size == 0:
            return 0.0
        
        total_pixels = soil_mask.shape[0] * soil_mask.shape[1]
        soil_pixels = np.sum(soil_mask > 0)
        
        return soil_pixels / total_pixels
    
    def _detect_reflection_areas(self, image: np.ndarray) -> float:
        """检测反光区域比例"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测高亮区域（反光）
        _, reflection_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_OPEN, kernel)
        
        total_pixels = reflection_mask.shape[0] * reflection_mask.shape[1]
        reflection_pixels = np.sum(reflection_mask > 0)
        
        return reflection_pixels / total_pixels
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """计算图像对比度"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算标准差作为对比度指标
        contrast = np.std(gray)
        
        return float(contrast)
    
    def _detect_shadow_areas(self, image: np.ndarray) -> float:
        """检测阴影区域比例"""
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 检测低亮度区域（阴影）
        v_channel = hsv[:, :, 2]
        _, shadow_mask = cv2.threshold(v_channel, 50, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        total_pixels = shadow_mask.shape[0] * shadow_mask.shape[1]
        shadow_pixels = np.sum(shadow_mask > 0)
        
        return shadow_pixels / total_pixels
    
    def _calculate_mask_connectivity(self, soil_mask: np.ndarray) -> float:
        """计算掩码连通性分数"""
        if np.sum(soil_mask) == 0:
            return 0.0
        
        # 标记连通组件
        labeled_mask = label(soil_mask > 0)
        regions = regionprops(labeled_mask)
        
        if not regions:
            return 0.0
        
        # 找到最大连通组件
        largest_region = max(regions, key=lambda r: r.area)
        largest_area = largest_region.area
        total_soil_area = np.sum(soil_mask > 0)
        
        # 连通性分数 = 最大连通组件面积 / 总土壤面积
        connectivity_score = largest_area / total_soil_area
        
        return connectivity_score
    
    def _calculate_overall_quality_score(self, metrics: Dict) -> float:
        """计算综合质量分数（0-1）"""
        scores = []
        
        # 土壤覆盖率分数
        soil_score = min(metrics['soil_coverage'] / self.min_soil_coverage, 1.0)
        scores.append(soil_score)
        
        # 反光区域分数（越少越好）
        reflection_score = max(0, 1 - metrics['reflection_ratio'] / self.max_reflection_ratio)
        scores.append(reflection_score)
        
        # 对比度分数
        contrast_score = min(metrics['contrast_score'] / self.min_contrast, 1.0)
        scores.append(contrast_score)
        
        # 阴影区域分数（越少越好）
        shadow_score = max(0, 1 - metrics['shadow_ratio'] / self.max_shadow_ratio)
        scores.append(shadow_score)
        
        # 连通性分数
        connectivity_score = metrics['connectivity_score']
        scores.append(connectivity_score)
        
        # 加权平均
        weights = [0.25, 0.2, 0.2, 0.2, 0.15]  # 各指标权重
        
        overall_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return overall_score
    
    def create_quality_report(self, results: Dict) -> str:
        """创建质量评估报告"""
        if not self.enable_filtering:
            return "质量过滤已禁用"
        
        report_lines = []
        
        # 整体评估
        if results['is_good_quality']:
            report_lines.append("✅ 图像质量: 良好")
        else:
            report_lines.append("❌ 图像质量: 不合格")
        
        report_lines.append(f"质量分数: {results['quality_score']:.2f}")
        
        # 详细指标
        metrics = results['metrics']
        report_lines.append("\\n详细指标:")
        report_lines.append(f"- 土壤覆盖率: {metrics['soil_coverage']:.2%}")
        report_lines.append(f"- 反光区域比例: {metrics['reflection_ratio']:.2%}")
        report_lines.append(f"- 对比度: {metrics['contrast_score']:.1f}")
        report_lines.append(f"- 阴影区域比例: {metrics['shadow_ratio']:.2%}")
        report_lines.append(f"- 掩码连通性: {metrics['connectivity_score']:.2%}")
        
        # 问题列表
        if results['issues']:
            report_lines.append("\\n发现的问题:")
            for issue in results['issues']:
                report_lines.append(f"- {issue}")
        
        return "\\n".join(report_lines)
    
    def filter_low_quality_images(self, image_results: list) -> Tuple[list, list]:
        """
        过滤低质量图像
        
        Args:
            image_results: 图像处理结果列表
            
        Returns:
            Tuple[list, list]: (高质量结果, 低质量结果)
        """
        if not self.enable_filtering:
            return image_results, []
        
        high_quality = []
        low_quality = []
        
        for result in image_results:
            if result.get('status') == 'failed':
                low_quality.append(result)
                continue
            
            # 评估质量
            quality_assessment = self.assess_image_quality(
                result['original_image'],
                result['soil_mask']
            )
            
            # 添加质量信息到结果中
            result['quality_assessment'] = quality_assessment
            
            if quality_assessment['is_good_quality']:
                high_quality.append(result)
            else:
                low_quality.append(result)
        
        return high_quality, low_quality