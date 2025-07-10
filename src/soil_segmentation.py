import cv2
import numpy as np
from typing import List, Tuple, Optional

class SoilSegmentation:
    """土壤区域分割器 - 分离土壤区域并移除非土壤物体"""
    
    def __init__(self):
        pass
        
    
    def segment_soil_area(self, image: np.ndarray) -> np.ndarray:
        """
        基于排除法分割土壤区域：去除天空、植被、米尺等非土壤区域
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 土壤区域掩码
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建初始掩码（全部为土壤）
        soil_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # 1. 排除天空区域（蓝色，通常在上部）
        sky_mask = self._detect_sky_region_enhanced(image)
        soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(sky_mask))
        
        # 2. 排除植被区域（绿色）
        vegetation_mask = self._detect_vegetation_region(hsv)
        soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(vegetation_mask))
        
        # 3. 排除米尺区域（黑白长条）
        ruler_mask = self._detect_ruler_region(image)
        soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(ruler_mask))
        
        # 4. 排除极亮区域（过曝光）
        bright_mask = self._detect_overexposed_region(image)
        soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(bright_mask))
        
        # 5. 排除工具区域（底部竖直放置的小刀、锤子等）
        tools_mask = self._detect_tools_region(image)
        soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(tools_mask))
        
        # 6. 可选：排除纯黑色阴影区域（根据需要决定是否启用）
        # shadow_mask = self._detect_pure_black_shadows(image)
        # soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(shadow_mask))
        
        # 轻微的形态学操作去除小噪声（保留土壤阴影、根系和石头）
        kernel = np.ones((3, 3), np.uint8)
        soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_OPEN, kernel)
        soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
        
        return soil_mask
    
    def _detect_sky_region_enhanced(self, image: np.ndarray) -> np.ndarray:
        """增强的天空区域检测 - 重点过滤上半部分的天空和杂物"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # 1. 检测蓝色天空
        lower_blue_sky = np.array([100, 50, 80], dtype=np.uint8)
        upper_blue_sky = np.array([130, 255, 255], dtype=np.uint8)
        blue_sky_mask = cv2.inRange(hsv, lower_blue_sky, upper_blue_sky)
        
        # 2. 检测浅蓝色天空
        lower_light_blue = np.array([90, 30, 150], dtype=np.uint8)
        upper_light_blue = np.array([120, 200, 255], dtype=np.uint8)
        light_blue_mask = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
        
        # 3. 检测白色/灰色天空（阴天）
        lower_white_sky = np.array([0, 0, 200], dtype=np.uint8)
        upper_white_sky = np.array([180, 40, 255], dtype=np.uint8)
        white_sky_mask = cv2.inRange(hsv, lower_white_sky, upper_white_sky)
        
        # 4. 检测灰色天空
        lower_gray_sky = np.array([0, 0, 150], dtype=np.uint8)
        upper_gray_sky = np.array([180, 30, 200], dtype=np.uint8)
        gray_sky_mask = cv2.inRange(hsv, lower_gray_sky, upper_gray_sky)
        
        # 组合所有天空掩码
        sky_mask = cv2.bitwise_or(blue_sky_mask, light_blue_mask)
        sky_mask = cv2.bitwise_or(sky_mask, white_sky_mask)
        sky_mask = cv2.bitwise_or(sky_mask, gray_sky_mask)
        
        # 形态学操作
        kernel = np.ones((3, 3), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel)
        
        kernel = np.ones((7, 7), np.uint8)
        sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)
        
        # 主要在图像上半部分寻找天空，但也考虑中间部分
        sky_mask_filtered = np.zeros_like(sky_mask)
        sky_mask_filtered[:int(h*0.7), :] = sky_mask[:int(h*0.7), :]
        
        return sky_mask_filtered
    
    def _detect_vegetation_region(self, hsv: np.ndarray) -> np.ndarray:
        """检测植被区域（绿色） - 增强版"""
        # 多种绿色植被的HSV范围
        
        # 浅绿色植被
        lower_light_green = np.array([35, 30, 30], dtype=np.uint8)
        upper_light_green = np.array([85, 255, 255], dtype=np.uint8)
        light_green_mask = cv2.inRange(hsv, lower_light_green, upper_light_green)
        
        # 深绿色植被
        lower_dark_green = np.array([40, 50, 20], dtype=np.uint8)
        upper_dark_green = np.array([80, 255, 200], dtype=np.uint8)
        dark_green_mask = cv2.inRange(hsv, lower_dark_green, upper_dark_green)
        
        # 黄绿色植被（干草等）
        lower_yellow_green = np.array([25, 40, 40], dtype=np.uint8)
        upper_yellow_green = np.array([40, 255, 255], dtype=np.uint8)
        yellow_green_mask = cv2.inRange(hsv, lower_yellow_green, upper_yellow_green)
        
        # 合并所有植被掩码
        vegetation_mask = cv2.bitwise_or(light_green_mask, dark_green_mask)
        vegetation_mask = cv2.bitwise_or(vegetation_mask, yellow_green_mask)
        
        # 形态学操作去除噪声并填充空洞
        kernel = np.ones((3, 3), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
        
        kernel = np.ones((5, 5), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        
        return vegetation_mask
    
    def _detect_ruler_region(self, image: np.ndarray) -> np.ndarray:
        """基于形状和纹理特征检测米尺区域"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. 边缘检测 - 找到强边缘
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 2. 霍夫直线检测 - 寻找长直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                               minLineLength=100, maxLineGap=10)
        
        ruler_mask = np.zeros_like(gray)
        
        if lines is not None:
            # 3. 分析每条直线，寻找米尺候选
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # 只考虑长线段
                if length > 200:
                    # 计算线段方向
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    # 垂直或水平方向的长线段
                    if abs(angle) < 15 or abs(angle) > 165 or abs(abs(angle) - 90) < 15:
                        # 在线段周围创建矩形区域
                        thickness = 30  # 米尺厚度
                        
                        # 计算垂直于线段的方向
                        if abs(angle) < 15 or abs(angle) > 165:  # 水平线
                            # 创建水平矩形
                            y_start = max(0, min(y1, y2) - thickness//2)
                            y_end = min(h, max(y1, y2) + thickness//2)
                            x_start = max(0, min(x1, x2) - 10)
                            x_end = min(w, max(x1, x2) + 10)
                        else:  # 垂直线
                            # 创建垂直矩形
                            x_start = max(0, min(x1, x2) - thickness//2)
                            x_end = min(w, max(x1, x2) + thickness//2)
                            y_start = max(0, min(y1, y2) - 10)
                            y_end = min(h, max(y1, y2) + 10)
                        
                        # 提取候选区域
                        candidate_region = gray[y_start:y_end, x_start:x_end]
                        
                        if candidate_region.size > 0:
                            # 4. 检测刻度纹理
                            if self._has_ruler_texture(candidate_region):
                                ruler_mask[y_start:y_end, x_start:x_end] = 255
        
        # 5. 形态学操作连接相邻区域
        kernel = np.ones((5, 5), np.uint8)
        ruler_mask = cv2.morphologyEx(ruler_mask, cv2.MORPH_CLOSE, kernel)
        
        # 6. 最终形状过滤
        contours, _ = cv2.findContours(ruler_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(ruler_mask)
        
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            width, height = rect[1]
            if width > 0 and height > 0:
                aspect_ratio = max(width, height) / min(width, height)
                area = cv2.contourArea(contour)
                # 长条形且面积适中
                if aspect_ratio > 4 and 800 < area < 150000:
                    cv2.fillPoly(filtered_mask, [contour], 255)
        
        return filtered_mask
    
    def _has_ruler_texture(self, region: np.ndarray) -> bool:
        """检测区域是否具有米尺的刻度纹理特征"""
        if region.size == 0:
            return False
            
        h, w = region.shape
        
        # 检测垂直线条（刻度）
        edges = cv2.Canny(region, 30, 100)
        
        # 使用霍夫直线检测刻度线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10,
                               minLineLength=min(h, w)//4, maxLineGap=3)
        
        if lines is None:
            return False
        
        # 统计垂直线条数量
        vertical_lines = 0
        horizontal_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
            
            if abs(abs(angle) - 90) < 20:  # 接近垂直
                vertical_lines += 1
            elif abs(angle) < 20 or abs(angle) > 160:  # 接近水平
                horizontal_lines += 1
        
        # 米尺应该有多条垂直刻度线
        return vertical_lines >= 3 or (vertical_lines >= 2 and horizontal_lines >= 1)
    
    def _detect_overexposed_region(self, image: np.ndarray) -> np.ndarray:
        """检测过曝光区域"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测极亮区域
        _, bright_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        
        return bright_mask
    
    def _detect_tools_region(self, image: np.ndarray) -> np.ndarray:
        """检测工具区域（小刀、锤子等竖直放置的工具） - 主要在图像底部"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # 重点检测底部区域（工具最可能出现的地方）
        bottom_region_height = int(h * 0.4)  # 底部40%区域
        bottom_region = image[h - bottom_region_height:, :]
        bottom_hsv = hsv[h - bottom_region_height:, :]
        
        # 1. 检测金属工具的典型颜色
        
        # 银色/灰色金属（刀具、锤子头）
        lower_metal = np.array([0, 0, 120], dtype=np.uint8)
        upper_metal = np.array([180, 40, 220], dtype=np.uint8)
        metal_mask = cv2.inRange(bottom_hsv, lower_metal, upper_metal)
        
        # 暗灰色金属（鉄制工具）
        lower_dark_metal = np.array([0, 0, 60], dtype=np.uint8)
        upper_dark_metal = np.array([180, 60, 140], dtype=np.uint8)
        dark_metal_mask = cv2.inRange(bottom_hsv, lower_dark_metal, upper_dark_metal)
        
        # 黄色/棕色工具手柄
        lower_brown = np.array([8, 50, 50], dtype=np.uint8)
        upper_brown = np.array([25, 255, 200], dtype=np.uint8)
        brown_mask = cv2.inRange(bottom_hsv, lower_brown, upper_brown)
        
        # 黑色工具手柄
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([180, 255, 60], dtype=np.uint8)
        black_mask = cv2.inRange(bottom_hsv, lower_black, upper_black)
        
        # 组合所有工具颜色
        tools_mask = cv2.bitwise_or(metal_mask, dark_metal_mask)
        tools_mask = cv2.bitwise_or(tools_mask, brown_mask)
        tools_mask = cv2.bitwise_or(tools_mask, black_mask)
        
        # 2. 形态学操作去除噪声
        kernel = np.ones((3, 3), np.uint8)
        tools_mask = cv2.morphologyEx(tools_mask, cv2.MORPH_OPEN, kernel)
        
        kernel = np.ones((5, 5), np.uint8)
        tools_mask = cv2.morphologyEx(tools_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. 根据形状特征过滤工具候选区域
        contours, _ = cv2.findContours(tools_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_tools_mask = np.zeros_like(tools_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # 过滤小区域
                # 检查形状特征
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    # 工具通常有一定的长宽比（竖直放置的刀、锤子）
                    if 1.5 <= aspect_ratio <= 10:
                        # 检查是否竖直放置（高度大于宽度）
                        angle = rect[2]
                        if angle < -45:
                            angle += 90
                        
                        # 接近垂直放置的工具
                        if abs(angle) < 30:  # 垂直放置误差在±30度内
                            cv2.fillPoly(filtered_tools_mask, [contour], 255)
        
        # 4. 创建完整尺寸的掩码（只有底部区域有工具）
        full_tools_mask = np.zeros((h, w), dtype=np.uint8)
        full_tools_mask[h - bottom_region_height:, :] = filtered_tools_mask
        
        return full_tools_mask
    
    def _detect_pure_black_shadows(self, image: np.ndarray) -> np.ndarray:
        """检测纯黑色阴影区域（可选去除）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测非常暗的区域
        _, shadow_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
        
        # 形态学操作去除小噪声
        kernel = np.ones((5, 5), np.uint8)
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
        
        # 过滤极小区域
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_shadow_mask = np.zeros_like(shadow_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # 只保留较大的阴影区域
                cv2.fillPoly(filtered_shadow_mask, [contour], 255)
        
        return filtered_shadow_mask
    
    def _detect_vertical_tools_in_bottom(self, image: np.ndarray) -> np.ndarray:
        """检测底部竖直放置的工具（小刀、锤子等）"""
        h, w = image.shape[:2]
        
        # 只检测底部区域
        bottom_height = int(h * 0.3)
        bottom_region = image[h - bottom_height:, :]
        
        if bottom_region.size == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # 使用边缘检测找到工具轮廓
        edges = cv2.Canny(gray, 50, 150)
        
        # 形态学操作连接边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tools_mask = np.zeros_like(gray)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 <= area <= 20000:  # 工具大小范围
                # 检查形状特征
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    # 工具通常有一定的长宽比且竖直放置
                    if 2 <= aspect_ratio <= 8:
                        angle = rect[2]
                        if angle < -45:
                            angle += 90
                        
                        # 检查是否竖直放置
                        if abs(angle) < 25:  # 垂直放置误差在±25度内
                            cv2.fillPoly(tools_mask, [contour], 255)
        
        # 创建完整尺寸的掩码
        full_tools_mask = np.zeros((h, w), dtype=np.uint8)
        full_tools_mask[h - bottom_height:, :] = tools_mask
        
        return full_tools_mask
    
    
    
    def apply_mask(self, image: np.ndarray, mask: np.ndarray, 
                   mask_type: str = 'transparent') -> np.ndarray:
        """
        应用掩码到图像
        
        Args:
            image: 输入图像
            mask: 掩码图像
            mask_type: 掩码类型 ('transparent' 或 'black')
            
        Returns:
            np.ndarray: 处理后的图像
        """
        if mask_type == 'transparent':
            # 创建RGBA图像
            if image.shape[2] == 3:
                result = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            else:
                result = image.copy()
            
            # 设置透明度
            result[mask > 0, 3] = 0  # 完全透明
            
        elif mask_type == 'black':
            result = image.copy()
            result[mask > 0] = [0, 0, 0]  # 设置为黑色
            
        else:
            raise ValueError(f"Unsupported mask_type: {mask_type}")
        
        return result
    
    def process_image(self, image: np.ndarray, 
                     ruler_mask: Optional[np.ndarray] = None,
                     mask_type: str = 'transparent') -> dict:
        """
        完整的图像处理流程
        
        Args:
            image: 输入图像
            ruler_mask: 米尺区域掩码
            mask_type: 掩码类型
            
        Returns:
            dict: 处理结果
        """
        # 分割土壤区域
        soil_mask = self.segment_soil_area(image)
        
        # 创建去除掩码（非土壤区域）
        remove_mask = cv2.bitwise_not(soil_mask)
        
        # 应用掩码
        processed_image = self.apply_mask(image, remove_mask, mask_type)
        
        return {
            'processed_image': processed_image,
            'soil_mask': soil_mask,
            'remove_mask': remove_mask,
            'detected_objects': []  # Initialize empty list for detected objects
        }