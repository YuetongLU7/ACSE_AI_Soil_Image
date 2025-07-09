"""
工具函数模块
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import json
from pathlib import Path

def load_image(image_path: str) -> np.ndarray:
    """
    加载图像文件
    
    Args:
        image_path: 图像路径
        
    Returns:
        np.ndarray: 图像数组
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return image

def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    保存图像文件
    
    Args:
        image: 图像数组
        output_path: 输出路径
        
    Returns:
        bool: 是否保存成功
    """
    return cv2.imwrite(output_path, image)

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect_ratio: bool = True) -> np.ndarray:
    """
    调整图像大小
    
    Args:
        image: 输入图像
        target_size: 目标尺寸 (width, height)
        maintain_aspect_ratio: 是否保持宽高比
        
    Returns:
        np.ndarray: 调整后的图像
    """
    if maintain_aspect_ratio:
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        
        # 计算新的尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整大小
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建目标尺寸的画布
        canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        
        # 计算偏移量
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # 将调整后的图像放置在画布中心
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    else:
        return cv2.resize(image, target_size)

def calculate_image_stats(image: np.ndarray) -> dict:
    """
    计算图像统计信息
    
    Args:
        image: 输入图像
        
    Returns:
        dict: 统计信息
    """
    if len(image.shape) == 3:
        # 彩色图像
        stats = {
            'shape': image.shape,
            'mean': np.mean(image, axis=(0, 1)).tolist(),
            'std': np.std(image, axis=(0, 1)).tolist(),
            'min': np.min(image, axis=(0, 1)).tolist(),
            'max': np.max(image, axis=(0, 1)).tolist()
        }
    else:
        # 灰度图像
        stats = {
            'shape': image.shape,
            'mean': float(np.mean(image)),
            'std': float(np.std(image)),
            'min': float(np.min(image)),
            'max': float(np.max(image))
        }
    
    return stats

def create_visualization_grid(images: List[np.ndarray], 
                            titles: List[str] = None,
                            grid_size: Tuple[int, int] = None) -> np.ndarray:
    """
    创建图像网格用于可视化
    
    Args:
        images: 图像列表
        titles: 标题列表
        grid_size: 网格尺寸 (rows, cols)
        
    Returns:
        np.ndarray: 网格图像
    """
    if not images:
        return np.array([])
    
    num_images = len(images)
    
    if grid_size is None:
        # 自动计算网格尺寸
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size
    
    # 获取图像尺寸
    h, w = images[0].shape[:2]
    
    # 创建网格画布
    if len(images[0].shape) == 3:
        grid = np.zeros((rows * h, cols * w, images[0].shape[2]), dtype=images[0].dtype)
    else:
        grid = np.zeros((rows * h, cols * w), dtype=images[0].dtype)
    
    # 放置图像
    for i, image in enumerate(images):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        y_start = row * h
        y_end = y_start + h
        x_start = col * w
        x_end = x_start + w
        
        # 确保图像尺寸匹配
        if image.shape[:2] != (h, w):
            image = cv2.resize(image, (w, h))
        
        grid[y_start:y_end, x_start:x_end] = image
        
        # 添加标题
        if titles and i < len(titles):
            cv2.putText(grid, titles[i], (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return grid

def filter_contours_by_area(contours: List[np.ndarray], 
                           min_area: int = 100,
                           max_area: Optional[int] = None) -> List[np.ndarray]:
    """
    根据面积过滤轮廓
    
    Args:
        contours: 轮廓列表
        min_area: 最小面积
        max_area: 最大面积
        
    Returns:
        List[np.ndarray]: 过滤后的轮廓列表
    """
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area >= min_area:
            if max_area is None or area <= max_area:
                filtered_contours.append(contour)
    
    return filtered_contours

def create_mask_from_contours(contours: List[np.ndarray], 
                            image_shape: Tuple[int, int]) -> np.ndarray:
    """
    从轮廓创建掩码
    
    Args:
        contours: 轮廓列表
        image_shape: 图像形状 (height, width)
        
    Returns:
        np.ndarray: 掩码图像
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, contours, 255)
    return mask

def save_metadata(metadata: dict, output_path: str):
    """
    保存元数据到JSON文件
    
    Args:
        metadata: 元数据字典
        output_path: 输出路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def load_metadata(input_path: str) -> dict:
    """
    从JSON文件加载元数据
    
    Args:
        input_path: 输入路径
        
    Returns:
        dict: 元数据字典
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_image_files(directory: str, extensions: List[str] = None) -> List[Path]:
    """
    获取目录中的图像文件
    
    Args:
        directory: 目录路径
        extensions: 支持的扩展名列表
        
    Returns:
        List[Path]: 图像文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    directory_path = Path(directory)
    image_files = []
    
    for ext in extensions:
        image_files.extend(directory_path.glob(f'*{ext}'))
        image_files.extend(directory_path.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)