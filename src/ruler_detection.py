import cv2
import numpy as np
from typing import Tuple, List, Optional
import math
import re
try:
    import easyocr
    OCR_AVAILABLE = True
    print("Utilisation d'EasyOCR pour la détection des chiffres")
except ImportError:
    try:
        import pytesseract
        import os

        # Reset Tesseract path
        tesseract_paths = [
            r'E:\SoftwareForStudy\tesseract.exe',
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        ]
        
        tesseract_found = False
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                tesseract_found = True
                print(f"Tesseract trouvé: {path}")
                break
        
        OCR_AVAILABLE = tesseract_found
        if not tesseract_found:
            print("Avertissement: OCR engine non trouvé")
    except ImportError:
        OCR_AVAILABLE = False
        print("OCR engine non installé")

class RulerDetector:
    """Ruler Detector - Detect the ruler in the image and give the pixel/cm ratio"""
    
    def __init__(self, min_length: int = 100, max_length: int = 2000):
        self.min_length = min_length
        self.max_length = max_length
        self._ocr_reader = None  # 缓存OCR模型，避免重复初始化
        
    def detect_ruler(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect the ruler in the image - focus on digit detection approach
        
        Args:
            image: input image in BGR format
            
        Returns:
            dict: a dictionary containing the ruler information, if not detected return None
        """
        # Method: Detection with OCR
        ruler_info = self._detect_ruler_by_digits(image)
        
        if ruler_info is not None:
            return ruler_info
        
    def _detect_ruler_by_digits(self, image: np.ndarray) -> Optional[dict]:
        """
        通过OCR检测刻度数字来检测米尺
        
        Args:
            image: 输入图像
            
        Returns:
            dict: 米尺信息，如果检测失败返回None
        """
        if not OCR_AVAILABLE:
            return None
        
        # 检测刻度数字
        digits_info = self._detect_scale_digits_enhanced(image)
        
        if not digits_info or len(digits_info) < 2:
            return None
        
        # 计算像素/厘米比例
        scale_ratio = self._calculate_scale_ratio_from_digits(digits_info)
        
        if scale_ratio is None:
            return None
        
        # 找到数字的横坐标中位数
        x_coords = [digit['center_x'] for digit in digits_info]
        median_x = int(np.median(x_coords))
        
        # 计算数字区域的中位数坐标
        y_coords = [digit['center_y'] for digit in digits_info]
        median_y = int(np.median(y_coords))
        
        # 创建双重掩码：米尺区域 + 数字区域
        h, w = image.shape[:2]
        mask_left = max(0, median_x - 200)
        mask_right = min(w, median_x + 200)
        
        # 数字区域掩码范围
        digit_mask_left = max(0, median_x - 170)
        digit_mask_right = min(w, median_x + 170)
        digit_mask_top = max(0, median_y - 250)
        digit_mask_bottom = min(h, median_y + 250)
        
        # 找到最顶部的数字作为深度参考
        top_digit = min(digits_info, key=lambda d: d['y'])
        
        # 创建简化的线坐标用于兼容现有接口
        line_coords = np.array([median_x, 0, median_x, h])
        
        return {
            'line_coords': line_coords,
            'pixel_length': h,
            'scale_ratio': scale_ratio,
            'ruler_detected': True,
            'confidence': 0.9,
            'detection_method': 'digit_detection',
            'median_x': median_x,
            'median_y': median_y,
            'mask_left': mask_left,
            'mask_right': mask_right,
            'digit_mask_left': digit_mask_left,
            'digit_mask_right': digit_mask_right,
            'digit_mask_top': digit_mask_top,
            'digit_mask_bottom': digit_mask_bottom,
            'top_digit_value': top_digit['value'],
            'top_digit_y': top_digit['y'],
            'detected_digits': digits_info
        }
    
    def _detect_scale_digits_enhanced(self, image: np.ndarray) -> List[dict]:
        """
        使用OCR检测米尺刻度数字 (0, 10, 20, 30, ..., 120)
        
        Args:
            image: 输入图像
            
        Returns:
            List[dict]: 检测到的数字信息列表
        """
        h, w = image.shape[:2]
        
        try:
            # 尝试使用 EasyOCR - 使用缓存的模型
            if self._ocr_reader is None:
                # 只在第一次使用时初始化
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            
            try:
                results = self._ocr_reader.readtext(image, allowlist='0123456789')
            except Exception as ocr_error:
                print(f"EasyOCR failed for this image: {ocr_error}")
                return []
            
            valid_digits = []
            for (bbox, text, confidence) in results:
                if text.isdigit() and confidence > 0.5:
                    num = int(text)
                    # 只保留米尺刻度：10的倍数，0-120cm
                    if 0 <= num <= 120 and num % 10 == 0:
                        # 计算边界框中心
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x = int(min(x_coords))
                        y = int(min(y_coords))
                        w = int(max(x_coords) - min(x_coords))
                        h = int(max(y_coords) - min(y_coords))
                        
                        valid_digits.append({
                            'value': num,
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'center_x': x + w//2,
                            'center_y': y + h//2,
                            'confidence': confidence
                        })
            
            # 去除重复
            valid_digits = self._remove_duplicate_digits(valid_digits)
            valid_digits.sort(key=lambda d: d['y'])
            
            if valid_digits:
                print(f"Traitement des chiffres détectés: {[d['value'] for d in valid_digits]}")

            return valid_digits
            
        except Exception as e:
            print(f"Erreur de détection des chiffres: {e}")
            return []
    
    def _remove_duplicate_digits(self, digits: List[dict]) -> List[dict]:
        """去除重复检测的数字"""
        if not digits:
            return []
        
        # 按数值分组
        digit_groups = {}
        for digit in digits:
            value = digit['value']
            if value not in digit_groups:
                digit_groups[value] = []
            digit_groups[value].append(digit)
        
        # 对每个数值，保留置信度最高的检测
        filtered_digits = []
        for value, group in digit_groups.items():
            if len(group) == 1:
                filtered_digits.append(group[0])
            else:
                # 保留置信度最高的
                best_digit = max(group, key=lambda d: d['confidence'])
                filtered_digits.append(best_digit)
        
        return filtered_digits
    
    def _calculate_scale_ratio_from_digits(self, digits: List[dict]) -> Optional[float]:
        """
        从检测到的数字计算像素/厘米比例
        
        Args:
            digits: 检测到的数字信息列表
            
        Returns:
            float: 像素/厘米比例，如果计算失败返回None
        """
        if len(digits) < 2:
            return None
        
        # 按Y坐标排序
        sorted_digits = sorted(digits, key=lambda d: d['center_y'])
        
        # 寻找最佳的数字对来计算比例
        best_ratio = None
        max_distance = 0
        
        for i in range(len(sorted_digits)):
            for j in range(i+1, len(sorted_digits)):
                digit1 = sorted_digits[i]
                digit2 = sorted_digits[j]
                
                # 计算像素距离
                pixel_distance = abs(digit2['center_y'] - digit1['center_y'])
                
                # 计算厘米距离
                cm_distance = abs(digit2['value'] - digit1['value'])
                
                if cm_distance > 0 and pixel_distance > max_distance:
                    max_distance = pixel_distance
                    best_ratio = pixel_distance / cm_distance
        
        if best_ratio is not None:
            print(f"Calculer le rapport: {best_ratio:.2f} pixels/cm")

        return best_ratio
    
    def extract_ruler_region(self, image: np.ndarray, ruler_info: dict) -> np.ndarray:
        """从图像中提取米尺区域用于后续处理"""
        if not ruler_info['ruler_detected']:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 创建掩码，标记米尺区域
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 如果是数字检测方法，使用双重掩码保证
        if ruler_info.get('detection_method') == 'digit_detection':
            h, w = image.shape[:2]
            
            # 1. 米尺区域掩码（垂直条状）
            left = ruler_info['mask_left']
            right = ruler_info['mask_right']
            mask[:, left:right] = 255
            
            # 2. 数字区域掩码（矩形区域）
            digit_left = ruler_info['digit_mask_left']
            digit_right = ruler_info['digit_mask_right']
            digit_top = ruler_info['digit_mask_top']
            digit_bottom = ruler_info['digit_mask_bottom']
            mask[digit_top:digit_bottom, digit_left:digit_right] = 255
        else:
            # 使用原有的线条掩码方法
            x1, y1, x2, y2 = ruler_info['line_coords']
            thickness = 20
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        
        return mask
    
    def visualize_detection(self, image: np.ndarray, ruler_info: dict) -> np.ndarray:
        """可视化米尺检测结果"""
        if not ruler_info['ruler_detected']:
            return image
        
        result = image.copy()
        
        # 如果是数字检测方法，显示检测到的数字
        if ruler_info.get('detection_method') == 'digit_detection':
            # 绘制检测到的数字
            for digit in ruler_info.get('detected_digits', []):
                x, y, w, h = digit['x'], digit['y'], digit['width'], digit['height']
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result, str(digit['value']), (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 绘制中位线
            median_x = ruler_info['median_x']
            cv2.line(result, (median_x, 0), (median_x, image.shape[0]), (255, 0, 0), 2)
            
            # 绘制米尺掩码区域（红色）
            left = ruler_info['mask_left']
            right = ruler_info['mask_right']
            cv2.rectangle(result, (left, 0), (right, image.shape[0]), (0, 0, 255), 2)
            
            # 绘制数字掩码区域（蓝色）
            digit_left = ruler_info['digit_mask_left']
            digit_right = ruler_info['digit_mask_right']
            digit_top = ruler_info['digit_mask_top']
            digit_bottom = ruler_info['digit_mask_bottom']
            cv2.rectangle(result, (digit_left, digit_top), (digit_right, digit_bottom), (255, 0, 0), 2)
            
            # 添加深度参考信息
            cv2.putText(result, f"Top: {ruler_info['top_digit_value']}cm", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # 使用原有的线条可视化
            x1, y1, x2, y2 = ruler_info['line_coords']
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 添加比例信息
        cv2.putText(result, f"Scale: {ruler_info['scale_ratio']:.2f} px/cm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return result