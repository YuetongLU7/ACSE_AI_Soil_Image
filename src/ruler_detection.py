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
        
        # 计算基于数字框大小的动态掩码
        h, w = image.shape[:2]
        
        # 计算平均数字框宽度
        avg_digit_width = np.mean([digit['width'] for digit in digits_info])
        
        # 中轴线左右各一个数字框宽度的米尺区域
        mask_left = max(0, median_x - int(avg_digit_width))
        mask_right = min(w, median_x + int(avg_digit_width))
        
        # 为每个数字创建1.5倍长宽的个别掩码区域
        digit_mask_left = w
        digit_mask_right = 0
        digit_mask_top = h
        digit_mask_bottom = 0
        
        for digit in digits_info:
            # 计算每个数字的扩展区域：宽2倍，长3倍
            expand_w = int(digit['width'] * 2)
            expand_h = int(digit['height'] * 3)
            
            left = max(0, digit['center_x'] - expand_w // 2)
            right = min(w, digit['center_x'] + expand_w // 2)
            top = max(0, digit['center_y'] - expand_h // 2)
            bottom = min(h, digit['center_y'] + expand_h // 2)
            
            # 更新总体数字掩码区域
            digit_mask_left = min(digit_mask_left, left)
            digit_mask_right = max(digit_mask_right, right)
            digit_mask_top = min(digit_mask_top, top)
            digit_mask_bottom = max(digit_mask_bottom, bottom)
        
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
            'avg_digit_width': avg_digit_width,
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
                print(f"Échec EasyOCR pour cette image: {ocr_error}")
                return []
            
            valid_digits = []
            for (bbox, text, confidence) in results:
                if text.isdigit() and confidence > 0.5:
                    num = int(text)
                    # 只保留米尺刻度：10的倍数，0-120cm
                    if 0 <= num <= 200 and num % 10 == 0:
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
                print(f"Chiffres détectés: {[d['value'] for d in valid_digits]}")

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
            print(f"Calcul du rapport: {best_ratio:.2f} pixels/cm")

        return best_ratio
    
    def calculate_upper_boundary(self, ruler_info: dict) -> Optional[int]:
        """
        根据米尺0刻度位置计算上边界，用于去除植被
        
        Args:
            ruler_info: 米尺检测信息
            
        Returns:
            int: 上边界的y坐标，如果无法计算则返回None
        """
        if not ruler_info or not ruler_info.get('ruler_detected', False):
            return None
            
        detected_digits = ruler_info.get('detected_digits', [])
        if not detected_digits:
            return None
            
        # 查找是否直接检测到了0刻度
        zero_digit = None
        for digit in detected_digits:
            if digit['value'] == 0:
                zero_digit = digit
                break
        
        if zero_digit is not None:
            # 如果检测到0刻度，使用矩形框的上边界
            upper_boundary = zero_digit['y']
            print(f"Zéro détecté, utilisation de la limite supérieure: y={upper_boundary}")
            return upper_boundary
        else:
            # 如果没有检测到0刻度，根据已知刻度推断0的位置
            scale_ratio = ruler_info.get('scale_ratio')
            if scale_ratio is None:
                return None
                
            # 找到最顶部的刻度作为参考
            top_digit = min(detected_digits, key=lambda d: d['center_y'])
            top_value = top_digit['value']
            top_center_y = top_digit['center_y']
            
            # 推断0刻度的中心位置
            inferred_zero_y = top_center_y - (top_value * scale_ratio)
            
            # 由于0刻度印在刻度下方，直接使用推断的坐标作为上边界
            upper_boundary = int(inferred_zero_y)
            print(f"Position zéro inférée, utilisation des coordonnées: y={upper_boundary} (basé sur la graduation {top_value}cm)")
            return upper_boundary
    
    def extract_ruler_region(self, image: np.ndarray, ruler_info: dict) -> np.ndarray:
        """从图像中提取米尺区域用于后续处理"""
        if not ruler_info['ruler_detected']:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 创建掩码，标记米尺区域
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 如果是数字检测方法，使用双重掩码保证
        if ruler_info.get('detection_method') == 'digit_detection':
            h, w = image.shape[:2]
            
            # 1. 中轴线米尺区域掩码（垂直条状，基于平均数字宽度）
            left = ruler_info['mask_left']
            right = ruler_info['mask_right']
            mask[:, left:right] = 255
            
            # 2. 每个数字的扩展掩码：宽2倍，长3倍（防止斜尺子时中轴线偏移）
            detected_digits = ruler_info.get('detected_digits', [])
            for digit in detected_digits:
                expand_w = int(digit['width'] * 2)
                expand_h = int(digit['height'] * 3)
                
                digit_left = max(0, digit['center_x'] - expand_w // 2)
                digit_right = min(w, digit['center_x'] + expand_w // 2)
                digit_top = max(0, digit['center_y'] - expand_h // 2)
                digit_bottom = min(h, digit['center_y'] + expand_h // 2)
                
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
            
            # 绘制中轴线米尺掩码区域（红色）
            left = ruler_info['mask_left']
            right = ruler_info['mask_right']
            cv2.rectangle(result, (left, 0), (right, image.shape[0]), (0, 0, 255), 2)
            
            # 绘制每个数字的扩展掩码区域：宽2倍，长3倍（蓝色）
            for digit in ruler_info.get('detected_digits', []):
                expand_w = int(digit['width'] * 2)
                expand_h = int(digit['height'] * 3)
                
                digit_left = max(0, digit['center_x'] - expand_w // 2)
                digit_right = min(image.shape[1], digit['center_x'] + expand_w // 2)
                digit_top = max(0, digit['center_y'] - expand_h // 2)
                digit_bottom = min(image.shape[0], digit['center_y'] + expand_h // 2)
                
                cv2.rectangle(result, (digit_left, digit_top), (digit_right, digit_bottom), (255, 0, 0), 1)
            
            # 添加深度参考信息
            cv2.putText(result, f"Top: {ruler_info['top_digit_value']}cm", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 绘制上边界线（如果可以计算）
            upper_boundary = self.calculate_upper_boundary(ruler_info)
            if upper_boundary is not None:
                cv2.line(result, (0, upper_boundary), (image.shape[1], upper_boundary), (255, 255, 0), 3)
                cv2.putText(result, f"Limite sup.: y={upper_boundary}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            # 使用原有的线条可视化
            x1, y1, x2, y2 = ruler_info['line_coords']
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 添加比例信息
        cv2.putText(result, f"Scale: {ruler_info['scale_ratio']:.2f} px/cm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return result