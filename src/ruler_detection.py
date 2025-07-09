import cv2
import numpy as np
from typing import Tuple, List, Optional
import math
import re
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class RulerDetector:
    """Ruler Detector - Detect the ruler in the image and give the pixel/cm ratio"""
    
    def __init__(self, min_length: int = 100, max_length: int = 2000):
        self.min_length = min_length
        self.max_length = max_length
        
    def detect_ruler(self, image: np.ndarray) -> Optional[dict]:
        """
        Detect the ruler in the image - focus on vertical rectangular shape with scale marks
        
        Args:
            image: input image in BGR format
            
        Returns:
            dict: a dictionary containing the ruler information, if not detected return None
        """
        # 主要方法: 垂直长矩形形态检测 + 刻度识别
        ruler_info_morphology = self._detect_ruler_by_morphology(image)
        
        # 备用方法: 改进的霍夫直线检测（针对垂直方向）
        ruler_info_hough = self._detect_ruler_by_vertical_hough(image)
        
        # 选择最佳结果
        candidates = [ruler_info_morphology, ruler_info_hough]
        candidates = [c for c in candidates if c is not None]
        
        if not candidates:
            return None
        
        # 选择置信度最高的结果
        best_candidate = max(candidates, key=lambda x: x.get('confidence', 0))
        
        return best_candidate
    
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
            return image
            
        # 创建掩码，标记米尺区域
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        x1, y1, x2, y2 = ruler_info['line_coords']
        
        # 创建粗化的线条掩码
        thickness = 20
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
        
        return mask
    
    def visualize_detection(self, image: np.ndarray, ruler_info: dict) -> np.ndarray:
        """可视化米尺检测结果"""
        if not ruler_info['ruler_detected']:
            return image
            
        result = image.copy()
        x1, y1, x2, y2 = ruler_info['line_coords']
        
        # 绘制检测到的米尺
        cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # 添加文本信息
        cv2.putText(result, f"Scale: {ruler_info['scale_ratio']:.2f} px/cm", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return result