import cv2
import numpy as np
from typing import Tuple, List, Optional
import math
import re
try:
    import pytesseract
    import os
    
    # 设置 Tesseract 路径
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
            print(f"找到 Tesseract: {path}")
            break
    
    if not tesseract_found:
        print("警告: 未找到 Tesseract 可执行文件")
        print("请确保 Tesseract 已正确安装并设置路径")
    
    TESSERACT_AVAILABLE = tesseract_found
except ImportError:
    TESSERACT_AVAILABLE = False
    print("pytesseract 未安装")

class RulerDetector:
    """Ruler Detector - Detect the ruler in the image and give the pixel/cm ratio"""
    
    def __init__(self, min_length: int = 100, max_length: int = 2000):
        self.min_length = min_length
        self.max_length = max_length
        
    def detect_ruler(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect the ruler in the image - focus on digit detection approach
        
        Args:
            image: input image in BGR format
            
        Returns:
            dict: a dictionary containing the ruler information, if not detected return None
        """
        # 主要方法: 直接通过OCR检测数字
        ruler_info = self._detect_ruler_by_digits(image)
        
        if ruler_info is not None:
            return ruler_info
        
        # 备用方法: 原有的形态学检测
        ruler_info_morphology = self._detect_ruler_by_morphology(image)
        
        if ruler_info_morphology is not None:
            return ruler_info_morphology
        
        return None
    
    def _detect_ruler_by_digits(self, image: np.ndarray) -> Optional[dict]:
        """
        通过OCR检测刻度数字来检测米尺
        
        Args:
            image: 输入图像
            
        Returns:
            dict: 米尺信息，如果检测失败返回None
        """
        if not TESSERACT_AVAILABLE:
            print("Warning: pytesseract not available, cannot detect ruler digits")
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
        
        # 创建米尺掩码坐标（中位数左右200px）
        h, w = image.shape[:2]
        mask_left = max(0, median_x - 200)
        mask_right = min(w, median_x + 200)
        
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
            'mask_left': mask_left,
            'mask_right': mask_right,
            'top_digit_value': top_digit['value'],
            'top_digit_y': top_digit['y'],
            'detected_digits': digits_info
        }
    
    def _detect_scale_digits_enhanced(self, image: np.ndarray) -> List[dict]:
        """
        增强的刻度数字检测
        
        Args:
            image: 输入图像
            
        Returns:
            List[dict]: 检测到的数字信息列表
        """
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # 应用高斯滤波减少噪声
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # 自适应阈值处理
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 形态学操作去除小噪声
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 放大图像提高OCR准确性
        height, width = cleaned.shape
        scale_factor = max(3, 500 // min(height, width))  # 提高放大倍数
        resized = cv2.resize(cleaned, (width * scale_factor, height * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)
        
        try:
            # OCR配置 - 只检测数字
            config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            
            # 获取文本和位置数据
            data = pytesseract.image_to_data(resized, config=config, output_type=pytesseract.Output.DICT)
            
            # 提取有效的刻度数字
            valid_digits = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text.isdigit():
                    num = int(text)
                    # 只保留10的倍数且在合理范围内 (0-120cm)
                    if 0 <= num <= 120 and num % 10 == 0:
                        # 将坐标转换回原图像尺寸
                        x = data['left'][i] // scale_factor
                        y = data['top'][i] // scale_factor
                        w = data['width'][i] // scale_factor
                        h = data['height'][i] // scale_factor
                        confidence = data['conf'][i]
                        
                        # 过滤置信度太低的检测
                        if confidence > 30 and w > 3 and h > 3:
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
            
            # 去除重复检测
            valid_digits = self._remove_duplicate_digits(valid_digits)
            
            # 按Y坐标排序（从上到下）
            valid_digits.sort(key=lambda d: d['y'])
            
            print(f"检测到 {len(valid_digits)} 个刻度数字: {[d['value'] for d in valid_digits]}")
            
            return valid_digits
            
        except Exception as e:
            print(f"OCR检测错误: {e}")
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
            print(f"计算得到比例: {best_ratio:.2f} 像素/厘米")
        
        return best_ratio
    
    def _detect_ruler_by_morphology(self, image: np.ndarray) -> Optional[dict]:
        """使用形态学方法检测垂直长矩形卷尺"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用自适应阈值处理
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 形态学操作 - 闭运算连接断开的部分
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))  # 垂直核
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 查找最佳卷尺候选
        best_ruler = self._find_best_vertical_ruler(contours, image)
        
        if best_ruler is None:
            return None
        
        # 获取卷尺的中心线
        ruler_line = self._extract_ruler_centerline(best_ruler)
        
        if ruler_line is None:
            return None
        
        # 通过刻度识别计算像素/厘米比例
        scale_ratio = self._calculate_scale_ratio_from_marks(image, ruler_line, best_ruler)
        
        pixel_length = self._calculate_line_length(ruler_line)
        
        return {
            'line_coords': ruler_line,
            'pixel_length': pixel_length,
            'scale_ratio': scale_ratio,
            'ruler_detected': True,
            'confidence': 0.8,
            'detection_method': 'morphology'
        }
    
    def _detect_ruler_by_vertical_hough(self, image: np.ndarray) -> Optional[dict]:
        """使用改进的霍夫直线检测垂直卷尺"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫直线检测 - 重点检测垂直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=self.min_length, maxLineGap=20)
        
        if lines is None:
            return None
            
        # 寻找最长的垂直线作为卷尺
        longest_vertical_line = self._find_longest_vertical_line(lines)
        
        if longest_vertical_line is None:
            return None
            
        # 计算卷尺长度（像素）
        pixel_length = self._calculate_line_length(longest_vertical_line)
        
        # 通过卷尺刻度识别计算像素/厘米比例
        scale_ratio = self._calculate_scale_ratio_from_marks(image, longest_vertical_line)
        
        return {
            'line_coords': longest_vertical_line,
            'pixel_length': pixel_length,
            'scale_ratio': scale_ratio,
            'ruler_detected': True,
            'confidence': 0.6,
            'detection_method': 'vertical_hough'
        }
    
    def _detect_ruler_by_template(self, image: np.ndarray) -> Optional[dict]:
        """基于模板匹配检测米尺（简化版本）"""
        # 这里可以添加更复杂的模板匹配逻辑
        # 目前返回None，表示未实现
        return None
    
    def _find_longest_vertical_line(self, lines: np.ndarray) -> Optional[np.ndarray]:
        """寻找最长的垂直线"""
        max_length = 0
        longest_line = None
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 检查是否接近垂直（角度在80-100度之间）
            if abs(x2 - x1) < 0.1:  # 完全垂直
                angle = 90
            else:
                angle = abs(math.degrees(math.atan((y2 - y1) / (x2 - x1))))
            
            if (80 <= angle <= 100) and (self.min_length <= length <= self.max_length) and length > max_length:
                max_length = length
                longest_line = line[0]
                
        return longest_line
    
    def _find_best_vertical_ruler(self, contours: List, image: np.ndarray) -> Optional[np.ndarray]:
        """从轮廓中找到最佳的垂直尺子候选"""
        best_contour = None
        max_score = 0
        
        for contour in contours:
            # 获取轮廓的边界矩形
            x, y, w, h = cv2.boundingRect(contour)
            
            # 检查是否为垂直长矩形
            if h < self.min_length or h > self.max_length:
                continue
            
            aspect_ratio = h / w
            if aspect_ratio < 5:  # 长宽比至少为5:1
                continue
            
            # 检查轮廓位置是否合理（不在图像边缘）
            img_h, img_w = image.shape[:2]
            if x < 10 or x + w > img_w - 10:
                continue
            
            # 计算轮廓的垂直度评分
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            if angle < -45:
                angle += 90
            
            # 角度越接近0度（垂直），分数越高
            angle_score = max(0, 1 - abs(angle) / 45)
            
            # 综合评分
            score = aspect_ratio * angle_score * h
            
            if score > max_score:
                max_score = score
                best_contour = contour
        
        return best_contour
    
    def _extract_ruler_centerline(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """从轮廓中提取尺子的中心线"""
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 找到最长的两条边的中点
        distances = []
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(box[i] - box[j])
                distances.append((dist, i, j))
        
        # 找到最长的边
        distances.sort(reverse=True)
        longest_dist, idx1, idx2 = distances[0]
        
        # 返回中心线的端点
        return np.array([box[idx1][0], box[idx1][1], box[idx2][0], box[idx2][1]])
    
    def _calculate_line_length(self, line: np.ndarray) -> float:
        """计算直线长度"""
        x1, y1, x2, y2 = line
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _calculate_scale_ratio_from_marks(self, image: np.ndarray, line: np.ndarray, contour: np.ndarray = None) -> float:
        """
        从刻度标记计算像素/厘米比例
        """
        pixel_length = self._calculate_line_length(line)
        
        # 方法1: 尝试识别刻度标记
        cm_length = self._detect_ruler_marks(image, line, contour)
        
        # 方法2: 如果无法识别刻度，使用智能估算
        if cm_length is None:
            cm_length = self._estimate_ruler_length_by_size(pixel_length)
        
        return pixel_length / cm_length
    
    def _detect_ruler_marks(self, image: np.ndarray, line: np.ndarray, contour: np.ndarray = None) -> Optional[float]:
        """检测垂直尺子上的数字标记"""
        x1, y1, x2, y2 = line.astype(int)
        
        # 创建尺子区域的掩码
        if contour is not None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            roi = cv2.bitwise_and(image, image, mask=mask)
        else:
            # 如果没有轮廓，创建一个矩形ROI
            ruler_width = 50
            roi = self._extract_ruler_roi(image, line, ruler_width)
            if roi is None:
                return None
        
        # 方法1: 尝试OCR识别数字
        if TESSERACT_AVAILABLE:
            cm_length = self._detect_ruler_numbers_ocr(roi, line)
            if cm_length is not None:
                return cm_length
        
        # 方法2: 如果OCR失败，使用智能估算
        pixel_length = self._calculate_line_length(line)
        return self._estimate_ruler_length_by_size(pixel_length)
    
    def _detect_ruler_numbers_ocr(self, roi: np.ndarray, line: np.ndarray) -> Optional[float]:
        """使用OCR识别卷尺上的数字（识别10的倍数：0,10,20,30...）"""
        # 转换为灰度图
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 预处理图像以提高OCR准确性
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_roi)
        
        # 二值化
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 去噪
        kernel = np.ones((2,2), np.uint8)
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 放大图像以提高OCR准确性
        height, width = denoised.shape
        scale_factor = max(2, 300 // min(height, width))
        resized = cv2.resize(denoised, (width * scale_factor, height * scale_factor), 
                           interpolation=cv2.INTER_CUBIC)
        
        try:
            # 使用OCR识别数字和位置
            config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
            
            # 获取数字和位置信息
            data = pytesseract.image_to_data(resized, config=config, output_type=pytesseract.Output.DICT)
            
            # 提取有效的数字和位置
            valid_marks = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text.isdigit():
                    num = int(text)
                    # 只保留10的倍数（0,10,20,30...）且在合理范围内
                    if 0 <= num <= 120 and num % 10 == 0:
                        # 计算在原图中的位置
                        x = data['left'][i] // scale_factor
                        y = data['top'][i] // scale_factor
                        w = data['width'][i] // scale_factor
                        h = data['height'][i] // scale_factor
                        
                        # 计算数字中心位置
                        center_y = y + h // 2
                        
                        valid_marks.append({
                            'value': num,
                            'position': center_y,
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h
                        })
            
            if len(valid_marks) >= 2:
                # 按位置排序
                valid_marks.sort(key=lambda x: x['position'])
                
                # 计算像素/厘米比例
                first_mark = valid_marks[0]
                last_mark = valid_marks[-1]
                
                pixel_distance = abs(last_mark['position'] - first_mark['position'])
                cm_distance = abs(last_mark['value'] - first_mark['value'])
                
                if cm_distance > 0:
                    # 保存详细的米尺信息
                    ruler_info = {
                        'marks': valid_marks,
                        'min_value': first_mark['value'],
                        'max_value': last_mark['value'],
                        'pixel_distance': pixel_distance,
                        'cm_distance': cm_distance,
                        'scale_ratio': pixel_distance / cm_distance
                    }
                    
                    # 将米尺信息保存到实例变量中
                    self.ruler_info = ruler_info
                    
                    # 返回总长度
                    return float(cm_distance)
                    
        except Exception as e:
            print(f"OCR error: {e}")
            
        return None
    
    def _estimate_ruler_length_by_size(self, pixel_length: float) -> float:
        """基于像素长度智能估算米尺长度"""
        # 根据常见米尺尺寸进行估算
        common_lengths = [15, 20, 25, 30, 50, 100]  # 常见米尺长度(cm)
        
        # 基于像素长度估算最可能的长度
        # 这里使用简单的启发式规则
        if pixel_length < 200:
            return 15.0
        elif pixel_length < 300:
            return 20.0
        elif pixel_length < 400:
            return 25.0
        elif pixel_length < 600:
            return 30.0
        elif pixel_length < 800:
            return 50.0
        else:
            return 100.0
    
    def _extract_ruler_roi(self, image: np.ndarray, line: np.ndarray, width: int) -> Optional[np.ndarray]:
        """提取米尺区域"""
        try:
            x1, y1, x2, y2 = line.astype(int)
            
            # 计算米尺的方向向量和垂直向量
            direction = np.array([x2 - x1, y2 - y1], dtype=float)
            length = np.linalg.norm(direction)
            if length == 0:
                return None
            
            direction = direction / length
            perpendicular = np.array([-direction[1], direction[0]])
            
            # 计算米尺区域的四个角点
            half_width = width // 2
            
            p1 = np.array([x1, y1]) + perpendicular * half_width
            p2 = np.array([x1, y1]) - perpendicular * half_width
            p3 = np.array([x2, y2]) - perpendicular * half_width
            p4 = np.array([x2, y2]) + perpendicular * half_width
            
            # 确保坐标在图像范围内
            h, w = image.shape[:2]
            points = np.array([p1, p2, p3, p4], dtype=np.int32)
            
            # 检查是否所有点都在图像内
            if (np.any(points[:, 0] < 0) or np.any(points[:, 0] >= w) or
                np.any(points[:, 1] < 0) or np.any(points[:, 1] >= h)):
                return None
            
            # 创建掩码并提取ROI
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            
            # 提取ROI
            roi = cv2.bitwise_and(image, image, mask=mask)
            
            return roi
            
        except Exception:
            return None
    
    def extract_ruler_region(self, image: np.ndarray, ruler_info: dict) -> np.ndarray:
        """从图像中提取米尺区域用于后续处理"""
        if not ruler_info['ruler_detected']:
            return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 创建掩码，标记米尺区域
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # 如果是数字检测方法，使用简化的矩形掩码
        if ruler_info.get('detection_method') == 'digit_detection':
            h, w = image.shape[:2]
            left = ruler_info['mask_left']
            right = ruler_info['mask_right']
            
            # 创建垂直条状掩码
            mask[:, left:right] = 255
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
            
            # 绘制掩码区域
            left = ruler_info['mask_left']
            right = ruler_info['mask_right']
            cv2.rectangle(result, (left, 0), (right, image.shape[0]), (0, 0, 255), 2)
            
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