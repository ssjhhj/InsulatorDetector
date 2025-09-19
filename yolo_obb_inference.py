#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-OBB 模型推理脚本
支持加载YOLO-OBB模型，选择图片文件夹，进行旋转边界框检测并可视化结果
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import math

# 添加ultralytics到路径
sys.path.append(str(Path(__file__).parent / "ultralytics"))

try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Colors
    from ultralytics.utils.ops import scale_boxes
except ImportError as e:
    print(f"导入ultralytics失败: {e}")
    print("请确保已安装ultralytics: pip install ultralytics")
    sys.exit(1)


class YOLOOBBInference:
    """YOLO-OBB推理类"""
    
    def __init__(self, model_path: str):
        """
        初始化YOLO-OBB推理器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.colors = Colors()
        self.current_image = None
        self.current_image_path = None
        self.results = None
        
        # 初始化模型
        self.load_model()
    
    def load_model(self):
        """加载YOLO-OBB模型"""
        try:
            print(f"正在加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def predict_image(self, image_path: str, conf_threshold: float = 0.25) -> np.ndarray:
        """
        对单张图片进行OBB检测
        
        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            
        Returns:
            检测结果
        """
        try:
            # 进行预测
            results = self.model(image_path, conf=conf_threshold, verbose=False)
            return results[0]
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def draw_rotated_boxes(self, image: np.ndarray, obb_data: np.ndarray, 
                          class_names: dict, conf_threshold: float = 0.25, 
                          line_thickness: int = 1) -> np.ndarray:
        """
        在图片上绘制旋转边界框
        
        Args:
            image: 输入图片
            obb_data: OBB检测结果 [x, y, w, h, angle, conf, cls]
            class_names: 类别名称字典
            conf_threshold: 置信度阈值
            line_thickness: 线条粗细
            
        Returns:
            绘制了边界框的图片
        """
        if obb_data is None or len(obb_data) == 0:
            return image
        
        # 创建图片副本
        img_with_boxes = image.copy()
        
        for detection in obb_data:
            if len(detection) < 7:
                continue
                
            x, y, w, h, angle, conf, cls = detection[:7]
            
            # 过滤低置信度检测
            if conf < conf_threshold:
                continue
            
            # 获取类别ID和名称
            cls_id = int(cls)
            cls_name = class_names.get(cls_id, f"Class {cls_id}")
            
            # 获取颜色
            color = self.colors(cls_id, True)
            
            # 计算旋转矩形的四个顶点
            rect_points = self._get_rotated_rect_points(x, y, w, h, angle)
            
            # 绘制旋转矩形
            cv2.polylines(img_with_boxes, [rect_points], True, color, line_thickness)
        
        return img_with_boxes
    
    def process_rotated_boxes(self, image: np.ndarray, obb_data: np.ndarray, 
                             conf_threshold: float = 0.25, 
                             show_mask: bool = True, 
                             mask_color: tuple = (0, 0, 255),
                             return_contours: bool = False,
                             sample_points: int = 200) -> tuple:
        """
        对检测框区域进行二值化处理并生成掩码
        
        Args:
            image: 输入图片
            obb_data: OBB检测结果 [x, y, w, h, angle, conf, cls]
            conf_threshold: 置信度阈值
            show_mask: 是否显示掩码
            mask_color: 掩码颜色 (B, G, R)
            return_contours: 是否返回轮廓点
            sample_points: 轮廓点采样数量
            
        Returns:
            如果return_contours=True: (处理后的图片, 轮廓点列表)
            否则: 处理后的图片
        """
        if obb_data is None or len(obb_data) == 0:
            if return_contours:
                return image, []
            return image
        
        # 创建图片副本
        processed_image = image.copy()
        all_contours = []  # 存储所有轮廓点
        
        for detection in obb_data:
            if len(detection) < 7:
                continue
                
            x, y, w, h, angle, conf, cls = detection[:7]
            
            # 过滤低置信度检测
            if conf < conf_threshold:
                continue
            
            # 计算旋转矩形的四个顶点
            rect_points = self._get_rotated_rect_points(x, y, w, h, angle)
            
            # 直接在旋转矩形区域内进行二值化处理
            binary_mask = self._binarize_rotated_region(image, rect_points)
            
            if show_mask:
                # 将二值化结果映射到掩码区域
                mask_3channel = np.zeros_like(image)
                mask_3channel[binary_mask > 0] = mask_color
                
                # 将处理结果叠加到原图
                processed_image = cv2.addWeighted(processed_image, 0.7, mask_3channel, 0.3, 0)
            
            # 如果需要返回轮廓点
            if return_contours:
                contour_points = self._extract_contour_points_from_binary_mask(
                    binary_mask, rect_points, sample_points
                )
                if contour_points is not None:
                    all_contours.append(contour_points)
        
        if return_contours:
            return processed_image, all_contours
        return processed_image
    
    def _binarize_rotated_region(self, image: np.ndarray, rect_points: np.ndarray) -> np.ndarray:
        """
        直接在旋转矩形区域内进行二值化处理
        
        Args:
            image: 输入图片
            rect_points: 旋转矩形的四个顶点
            
        Returns:
            二值化掩码（与原图同尺寸）
        """
        try:
            # 创建与原图同尺寸的掩码
            binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            
            # 创建旋转矩形的掩码
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [rect_points], 255)
            
            # 在掩码区域内进行二值化处理
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # 使用自适应阈值进行二值化
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 反转二值化结果，使较暗的区域为白色
            binary = cv2.bitwise_not(binary)
            
            # 形态学操作，去除噪声
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 只在旋转矩形区域内保留二值化结果
            binary_mask[mask > 0] = binary[mask > 0]
            
            return binary_mask
            
        except Exception as e:
            print(f"旋转区域二值化失败: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)
    
    def _extract_contour_points_from_binary_mask(self, binary_mask: np.ndarray, 
                                               rect_points: np.ndarray, 
                                               sample_points: int = 200) -> np.ndarray:
        """
        从二值化掩码中提取轮廓点
        
        Args:
            binary_mask: 二值化掩码
            rect_points: 旋转矩形顶点
            sample_points: 采样点数量
            
        Returns:
            轮廓点坐标数组 [N, 2]
        """
        try:
            # 查找轮廓
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 计算轮廓周长
            perimeter = cv2.arcLength(largest_contour, True)
            
            # 使用Douglas-Peucker算法简化轮廓
            epsilon = 0.02 * perimeter
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # 如果简化后的轮廓点太少，使用原始轮廓
            if len(simplified_contour) < 4:
                simplified_contour = largest_contour
            
            # 均匀采样轮廓点
            contour_points = self._sample_contour_points(simplified_contour, sample_points)
            
            return contour_points
            
        except Exception as e:
            print(f"从二值化掩码提取轮廓点失败: {e}")
            return None
    
    def _extract_rotated_roi(self, image: np.ndarray, rect_points: np.ndarray) -> np.ndarray:
        """
        提取旋转矩形区域（精确提取框内部分）
        
        Args:
            image: 输入图片
            rect_points: 旋转矩形的四个顶点
            
        Returns:
            提取的ROI区域
        """
        try:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(rect_points)
            
            # 确保坐标在图片范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return None
            
            # 提取ROI
            roi = image[y:y+h, x:x+w]
            
            # 创建旋转矩形的掩码
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # 将旋转矩形顶点转换到ROI坐标系
            roi_rect_points = rect_points - np.array([x, y])
            
            # 填充旋转矩形区域
            cv2.fillPoly(mask, [roi_rect_points], 255)
            
            # 应用掩码，只保留框内部分
            if len(roi.shape) == 3:
                # 彩色图像
                masked_roi = roi.copy()
                for c in range(roi.shape[2]):
                    masked_roi[:, :, c] = np.where(mask > 0, roi[:, :, c], 0)
            else:
                # 灰度图像
                masked_roi = np.where(mask > 0, roi, 0)
            
            return masked_roi
            
        except Exception as e:
            print(f"提取ROI失败: {e}")
            return None
    
    def _binarize_roi(self, roi: np.ndarray) -> np.ndarray:
        """
        对ROI进行二值化处理，获取颜色较深的区域
        
        Args:
            roi: 输入ROI区域
            
        Returns:
            二值化结果
        """
        try:
            # 转换为灰度图
            if len(roi.shape) == 3:
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = roi.copy()
            
            # 使用自适应阈值进行二值化，获取较暗的区域
            # 使用OTSU方法自动确定阈值
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 反转二值化结果，使较暗的区域为白色
            binary = cv2.bitwise_not(binary)
            
            # 形态学操作，去除噪声
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            return binary
            
        except Exception as e:
            print(f"二值化处理失败: {e}")
            return np.zeros_like(roi)
    
    def _overlay_binary_on_roi(self, image: np.ndarray, binary_roi: np.ndarray, rect_points: np.ndarray):
        """
        将二值化结果叠加到原图的ROI区域
        
        Args:
            image: 原图
            binary_roi: 二值化ROI
            rect_points: 旋转矩形顶点
        """
        try:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(rect_points)
            
            # 确保坐标在图片范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return
            
            # 调整二值化ROI大小
            binary_resized = cv2.resize(binary_roi, (w, h))
            
            # 创建三通道掩码
            mask_3channel = np.zeros((h, w, 3), dtype=np.uint8)
            mask_3channel[binary_resized > 0] = [0, 0, 255]  # 红色掩码
            
            # 将掩码叠加到原图
            image[y:y+h, x:x+w] = cv2.addWeighted(
                image[y:y+h, x:x+w], 0.8, 
                mask_3channel, 0.2, 0
            )
            
        except Exception as e:
            print(f"叠加二值化结果失败: {e}")
    
    
    
    
    
    
    def _overlay_processed_on_roi(self, image: np.ndarray, processed_roi: np.ndarray, 
                                 rect_points: np.ndarray):
        """
        将处理结果叠加到原图的ROI区域
        
        Args:
            image: 原图
            processed_roi: 处理后的ROI
            rect_points: 旋转矩形顶点
        """
        try:
            # 获取边界框
            x, y, w, h = cv2.boundingRect(rect_points)
            
            # 确保坐标在图片范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return
            
            # 调整处理结果大小
            processed_resized = cv2.resize(processed_roi, (w, h))
            
            # 使用红色显示二值化结果
            color = [0, 0, 255]  # 红色
            
            # 创建三通道掩码
            mask_3channel = np.zeros((h, w, 3), dtype=np.uint8)
            mask_3channel[processed_resized > 0] = color
            
            # 将掩码叠加到原图
            image[y:y+h, x:x+w] = cv2.addWeighted(
                image[y:y+h, x:x+w], 0.8, 
                mask_3channel, 0.2, 0
            )
            
        except Exception as e:
            print(f"叠加处理结果失败: {e}")
    
    
    def _sample_contour_points(self, contour: np.ndarray, sample_points: int) -> np.ndarray:
        """
        均匀采样轮廓点
        
        Args:
            contour: 轮廓点数组
            sample_points: 采样点数量
            
        Returns:
            采样后的轮廓点
        """
        try:
            # 将轮廓点展平
            contour_flat = contour.reshape(-1, 2)
            
            # 如果轮廓点数量已经小于等于采样数量，直接返回
            if len(contour_flat) <= sample_points:
                return contour_flat
            
            # 计算轮廓的累积弧长
            distances = np.sqrt(np.sum(np.diff(contour_flat, axis=0)**2, axis=1))
            cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
            
            # 计算总弧长
            total_length = cumulative_distances[-1]
            
            # 均匀采样
            sample_distances = np.linspace(0, total_length, sample_points, endpoint=False)
            
            # 插值得到采样点
            sampled_points = np.zeros((sample_points, 2))
            for i, dist in enumerate(sample_distances):
                # 找到距离对应的索引
                idx = np.searchsorted(cumulative_distances, dist, side='right') - 1
                idx = min(idx, len(contour_flat) - 2)
                
                # 线性插值
                if idx < len(contour_flat) - 1:
                    t = (dist - cumulative_distances[idx]) / (cumulative_distances[idx + 1] - cumulative_distances[idx])
                    sampled_points[i] = (1 - t) * contour_flat[idx] + t * contour_flat[idx + 1]
                else:
                    sampled_points[i] = contour_flat[-1]
            
            return sampled_points.astype(np.int32)
            
        except Exception as e:
            print(f"采样轮廓点失败: {e}")
            return contour_flat[:sample_points] if len(contour_flat) > sample_points else contour_flat
    
    def draw_contour_points(self, image: np.ndarray, contour_points_list: list, 
                           point_color: tuple = (0, 255, 0), 
                           point_radius: int = 3,
                           connect_points: bool = True) -> np.ndarray:
        """
        在图片上绘制轮廓点
        
        Args:
            image: 输入图片
            contour_points_list: 轮廓点列表
            point_color: 点颜色 (B, G, R)
            point_radius: 点半径
            connect_points: 是否连接点
            
        Returns:
            绘制了轮廓点的图片
        """
        result_image = image.copy()
        
        for contour_points in contour_points_list:
            if contour_points is None or len(contour_points) == 0:
                continue
            
            # 绘制点
            for point in contour_points:
                cv2.circle(result_image, tuple(point), point_radius, point_color, -1)
            
            # 连接点
            if connect_points and len(contour_points) > 1:
                for i in range(len(contour_points)):
                    start_point = tuple(contour_points[i])
                    end_point = tuple(contour_points[(i + 1) % len(contour_points)])
                    cv2.line(result_image, start_point, end_point, point_color, 1)
        
        return result_image
    
    def extract_and_binarize_roi(self, image: np.ndarray, obb_data: np.ndarray, 
                                conf_threshold: float = 0.25) -> list:
        """
        提取旋转框区域并进行二值化处理
        
        Args:
            image: 输入图片
            obb_data: OBB检测结果 [x, y, w, h, angle, conf, cls]
            conf_threshold: 置信度阈值
            
        Returns:
            二值化ROI图像列表
        """
        binary_roi_list = []
        
        if obb_data is None or len(obb_data) == 0:
            return binary_roi_list
        
        for detection in obb_data:
            if len(detection) < 7:
                continue
                
            x, y, w, h, angle, conf, cls = detection[:7]
            
            # 过滤低置信度检测
            if conf < conf_threshold:
                continue
            
            # 计算旋转矩形的四个顶点
            rect_points = self._get_rotated_rect_points(x, y, w, h, angle)
            
            # 提取旋转矩形区域
            roi = self._extract_rotated_roi(image, rect_points)
            
            if roi is not None and roi.size > 0:
                # 对ROI进行二值化处理
                binary_roi = self._binarize_roi(roi)
                
                # 创建三通道图像用于显示
                binary_roi_3channel = cv2.cvtColor(binary_roi, cv2.COLOR_GRAY2BGR)
                
                # 添加边框和标签
                binary_roi_with_info = self._add_roi_info(binary_roi_3channel, conf, cls, w, h)
                
                binary_roi_list.append({
                    'roi': binary_roi_with_info,
                    'bbox': (x, y, w, h, angle),
                    'conf': conf,
                    'cls': cls
                })
        
        return binary_roi_list
    
    def _add_roi_info(self, roi_image: np.ndarray, conf: float, cls: int, w: float, h: float) -> np.ndarray:
        """
        在ROI图像上添加信息标签
        
        Args:
            roi_image: ROI图像
            conf: 置信度
            cls: 类别
            w, h: 宽度和高度
            
        Returns:
            添加了信息的ROI图像
        """
        try:
            # 创建信息文本
            # info_text = f"Conf: {conf:.2f}, Class: {int(cls)}"
            # size_text = f"Size: {int(w)}x{int(h)}"
            
            # 添加背景
            cv2.rectangle(roi_image, (0, 0), (roi_image.shape[1], 30), (0, 0, 0), -1)
            
            # 添加文本
            # cv2.putText(roi_image, info_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            # cv2.putText(roi_image, size_text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return roi_image
            
        except Exception as e:
            print(f"添加ROI信息失败: {e}")
            return roi_image
    
    def display_binary_rois(self, binary_roi_list: list, max_cols: int = 4) -> np.ndarray:
        """
        将多个二值化ROI图像组合显示
        
        Args:
            binary_roi_list: 二值化ROI图像列表
            max_cols: 最大列数
            
        Returns:
            组合后的图像
        """
        if not binary_roi_list:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 计算网格布局
        num_rois = len(binary_roi_list)
        cols = min(max_cols, num_rois)
        rows = (num_rois + cols - 1) // cols
        
        # 获取单个ROI的尺寸
        roi_height, roi_width = binary_roi_list[0]['roi'].shape[:2]
        
        # 创建组合图像
        combined_height = rows * roi_height
        combined_width = cols * roi_width
        combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
        
        # 填充ROI图像
        for i, roi_data in enumerate(binary_roi_list):
            row = i // cols
            col = i % cols
            
            y_start = row * roi_height
            y_end = y_start + roi_height
            x_start = col * roi_width
            x_end = x_start + roi_width
            
            combined_image[y_start:y_end, x_start:x_end] = roi_data['roi']
        
        return combined_image
    
    def extract_roi_images(self, image: np.ndarray, obb_data: np.ndarray, 
                          conf_threshold: float = 0.25) -> list:
        """
        提取检测框内的纯净图像（只保存框内部分，背景为黑色）
        
        Args:
            image: 输入图片
            obb_data: OBB检测结果 [x, y, w, h, angle, conf, cls]
            conf_threshold: 置信度阈值
            
        Returns:
            ROI图像列表
        """
        roi_images = []
        
        if obb_data is None or len(obb_data) == 0:
            return roi_images
        
        for i, detection in enumerate(obb_data):
            if len(detection) < 7:
                continue
                
            x, y, w, h, angle, conf, cls = detection[:7]
            
            # 过滤低置信度检测
            if conf < conf_threshold:
                continue
            
            # 计算旋转矩形的四个顶点
            rect_points = self._get_rotated_rect_points(x, y, w, h, angle)
            
            # 提取旋转矩形区域（只保留框内部分）
            roi = self._extract_rotated_roi(image, rect_points)
            
            if roi is not None and roi.size > 0:
                # 只保存纯净的ROI图像，不添加任何标签或框
                roi_images.append({
                    'roi': roi,  # 纯净的ROI图像（只包含框内部分）
                    'bbox': (x, y, w, h, angle),
                    'conf': conf,
                    'cls': cls,
                    'index': i+1
                })
        
        return roi_images
    
    def _add_roi_info_to_image(self, roi_image: np.ndarray, conf: float, cls: int, index: int) -> np.ndarray:
        """
        在ROI图像上添加信息标签
        
        Args:
            roi_image: ROI图像
            conf: 置信度
            cls: 类别
            index: 索引
            
        Returns:
            添加了信息的ROI图像
        """
        try:
            # 创建信息文本
            info_text = f"#{index} - Conf: {conf:.2f}, Class: {int(cls)}"
            
            # 添加背景条
            cv2.rectangle(roi_image, (0, 0), (roi_image.shape[1], 25), (0, 0, 0), -1)
            
            # 添加文本
            cv2.putText(roi_image, info_text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return roi_image
            
        except Exception as e:
            print(f"添加ROI信息失败: {e}")
            return roi_image
    
    def save_roi_images_to_folder(self, roi_images: list, save_folder: str) -> bool:
        """
        将纯净的ROI图像保存到文件夹（不包含框和标签）
        
        Args:
            roi_images: ROI图像列表
            save_folder: 保存文件夹路径
            
        Returns:
            是否保存成功
        """
        try:
            import os
            from pathlib import Path
            
            # 创建保存文件夹
            Path(save_folder).mkdir(parents=True, exist_ok=True)
            
            saved_count = 0
            for roi_data in roi_images:
                # 生成简洁的文件名
                filename = f"roi_{roi_data['index']:03d}.jpg"
                filepath = os.path.join(save_folder, filename)
                
                # 保存纯净的ROI图像（不包含任何框或标签）
                cv2.imwrite(filepath, roi_data['roi'])
                saved_count += 1
                
                # 可选：同时保存一个带信息的版本用于参考
                info_filename = f"roi_{roi_data['index']:03d}_info.txt"
                info_filepath = os.path.join(save_folder, info_filename)
                with open(info_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"ROI #{roi_data['index']}\n")
                    f.write(f"置信度: {roi_data['conf']:.3f}\n")
                    f.write(f"类别: {int(roi_data['cls'])}\n")
                    f.write(f"边界框: x={roi_data['bbox'][0]:.1f}, y={roi_data['bbox'][1]:.1f}, w={roi_data['bbox'][2]:.1f}, h={roi_data['bbox'][3]:.1f}, angle={roi_data['bbox'][4]:.3f}\n")
            
            print(f"成功保存 {saved_count} 个纯净ROI图像到: {save_folder}")
            print(f"同时保存了 {saved_count} 个信息文件用于参考")
            return True
            
        except Exception as e:
            print(f"保存ROI图像失败: {e}")
            return False
    
    def _get_rotated_rect_points(self, x: float, y: float, w: float, h: float, angle: float) -> np.ndarray:
        """
        计算旋转矩形的四个顶点
        
        Args:
            x, y: 中心点坐标
            w, h: 宽度和高度
            angle: 旋转角度（弧度）
            
        Returns:
            四个顶点的坐标数组
        """
        # 计算矩形的四个角点（相对于中心点）
        half_w, half_h = w / 2, h / 2
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])
        
        # 旋转矩阵
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # 应用旋转
        rotated_corners = corners @ rotation_matrix.T
        
        # 平移到实际位置
        points = rotated_corners + np.array([x, y])
        
        return points.astype(np.int32)
    


class YOLOOBBGUI:
    """YOLO-OBB图形界面"""
    
    def __init__(self, model_path: str):
        """
        初始化GUI
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.inference = YOLOOBBInference(model_path)
        self.image_files = []
        self.current_index = 0
        self.conf_threshold = 0.25
        self.show_mask = True
        self.mask_color = (0, 0, 255)  # 红色掩码
        self.contour_points = []  # 存储轮廓点
        self.show_contours = False  # 是否显示轮廓点
        self.show_binary_roi = False  # 是否显示二值化区域
        self.binary_roi_images = []  # 存储二值化ROI图像
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("YOLO-OBB 检测工具")
        self.root.geometry("1200x800")
        
        # 创建界面
        self.create_widgets()
        
        # 加载图片文件夹
        self.load_image_folder()
    
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件夹选择按钮
        ttk.Button(control_frame, text="选择图片文件夹", 
                  command=self.load_image_folder).pack(side=tk.LEFT, padx=(0, 10))
        
        # 置信度阈值滑块
        ttk.Label(control_frame, text="置信度阈值:").pack(side=tk.LEFT, padx=(0, 5))
        self.conf_var = tk.DoubleVar(value=self.conf_threshold)
        conf_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, 
                              variable=self.conf_var, orient=tk.HORIZONTAL,
                              command=self.update_conf_threshold)
        conf_scale.pack(side=tk.LEFT, padx=(0, 10))
        
        # 置信度值显示
        self.conf_label = ttk.Label(control_frame, text=f"{self.conf_threshold:.2f}")
        self.conf_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # 检测按钮
        ttk.Button(control_frame, text="开始检测", 
                  command=self.detect_current_image).pack(side=tk.LEFT, padx=(0, 10))
        
        # 掩码处理按钮
        ttk.Button(control_frame, text="掩码处理", 
                  command=self.process_with_mask).pack(side=tk.LEFT, padx=(0, 10))
        
        # 二值化ROI按钮
        ttk.Button(control_frame, text="二值化ROI", 
                  command=self.show_binary_rois).pack(side=tk.LEFT, padx=(0, 10))
        
        # 显示掩码复选框
        self.show_mask_var = tk.BooleanVar(value=self.show_mask)
        ttk.Checkbutton(control_frame, text="显示掩码", 
                       variable=self.show_mask_var,
                       command=self.toggle_mask_display).pack(side=tk.LEFT, padx=(0, 10))
        
        
        # 显示轮廓点复选框
        self.show_contours_var = tk.BooleanVar(value=self.show_contours)
        ttk.Checkbutton(control_frame, text="显示轮廓点", 
                       variable=self.show_contours_var,
                       command=self.toggle_contours_display).pack(side=tk.LEFT, padx=(0, 10))
        
        # 显示二值化区域复选框
        self.show_binary_roi_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="显示二值化区域", 
                       variable=self.show_binary_roi_var,
                       command=self.toggle_binary_roi_display).pack(side=tk.LEFT, padx=(0, 10))
        
        # 保存结果按钮
        ttk.Button(control_frame, text="保存结果", 
                  command=self.save_result).pack(side=tk.LEFT, padx=(0, 10))
        
        # 保存框内图像按钮
        ttk.Button(control_frame, text="保存框内图像", 
                  command=self.save_roi_images).pack(side=tk.LEFT, padx=(0, 10))
        
        # 图片导航
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(nav_frame, text="上一张", 
                  command=self.prev_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(nav_frame, text="下一张", 
                  command=self.next_image).pack(side=tk.LEFT, padx=(0, 5))
        
        # 图片信息标签
        self.info_label = ttk.Label(nav_frame, text="")
        self.info_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # 图片显示区域
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建滚动条
        self.canvas = tk.Canvas(self.image_frame, bg="white")
        self.scrollbar_v = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_h = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scrollbar_v.set, xscrollcommand=self.scrollbar_h.set)
        
        # 布局滚动条和画布
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar_v.grid(row=0, column=1, sticky="ns")
        self.scrollbar_h.grid(row=1, column=0, sticky="ew")
        
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
    
    def load_image_folder(self):
        """加载图片文件夹"""
        folder_path = filedialog.askdirectory(title="选择包含图片的文件夹")
        if not folder_path:
            return
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图片文件
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(Path(folder_path).glob(f"*{ext}"))
            self.image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
        
        self.image_files = sorted(self.image_files)
        
        if not self.image_files:
            messagebox.showwarning("警告", "所选文件夹中没有找到图片文件!")
            return
        
        self.current_index = 0
        self.load_current_image()
        self.update_info()
    
    def load_current_image(self):
        """加载当前图片"""
        if not self.image_files:
            return
        
        image_path = self.image_files[self.current_index]
        try:
            # 加载图片
            image = cv2.imread(str(image_path))
            if image is None:
                messagebox.showerror("错误", f"无法加载图片: {image_path}")
                return
            
            self.inference.current_image = image
            self.inference.current_image_path = str(image_path)
            
            # 显示原图
            self.display_image(image)
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {e}")
    
    def detect_current_image(self):
        """检测当前图片"""
        if self.inference.current_image is None:
            messagebox.showwarning("警告", "请先加载图片!")
            return
        
        try:
            # 进行检测
            self.results = self.inference.predict_image(
                self.inference.current_image_path, 
                self.conf_threshold
            )
            
            if self.results is None:
                messagebox.showerror("错误", "检测失败!")
                return
            
            # 获取OBB数据
            obb_data = self.results.obb.data.cpu().numpy() if self.results.obb is not None else None
            
            # 绘制检测结果
            if obb_data is not None and len(obb_data) > 0:
                image_with_boxes = self.inference.draw_rotated_boxes(
                    self.inference.current_image, 
                    obb_data, 
                    self.results.names,
                    self.conf_threshold,
                    line_thickness=2  # 检测时使用中等粗细
                )
                self.display_image(image_with_boxes)
                
                # 显示检测数量
                num_detections = len(obb_data)
                messagebox.showinfo("检测完成", f"检测到 {num_detections} 个目标")
            else:
                messagebox.showinfo("检测完成", "未检测到任何目标")
                
        except Exception as e:
            messagebox.showerror("错误", f"检测失败: {e}")
    
    def process_with_mask(self):
        """对当前图片进行掩码处理"""
        if self.inference.current_image is None:
            messagebox.showwarning("警告", "请先加载图片!")
            return
        
        if self.results is None:
            messagebox.showwarning("警告", "请先进行检测!")
            return
        
        try:
            # 获取OBB数据
            obb_data = self.results.obb.data.cpu().numpy() if self.results.obb is not None else None
            
            if obb_data is None or len(obb_data) == 0:
                messagebox.showinfo("提示", "没有检测到目标，无法进行掩码处理!")
                return
            
            # 进行掩码处理
            if self.show_contours:
                processed_image, self.contour_points = self.inference.process_rotated_boxes(
                    self.inference.current_image,
                    obb_data,
                    self.conf_threshold,
                    self.show_mask,
                    self.mask_color,
                    return_contours=True,
                    sample_points=200
                )
            else:
                processed_image = self.inference.process_rotated_boxes(
                    self.inference.current_image,
                    obb_data,
                    self.conf_threshold,
                    self.show_mask,
                    self.mask_color
                )
                self.contour_points = []
            
            # 如果需要显示轮廓点，绘制轮廓点和检测框
            if self.show_contours and self.contour_points:
                # 绘制轮廓点
                processed_image = self.inference.draw_contour_points(
                    processed_image, 
                    self.contour_points,
                    point_color=(0, 255, 0),  # 绿色轮廓点
                    point_radius=2,
                    connect_points=True
                )
                
                # 绘制检测框（细线）
                obb_data = self.results.obb.data.cpu().numpy() if self.results.obb is not None else None
                if obb_data is not None and len(obb_data) > 0:
                    processed_image = self.inference.draw_rotated_boxes(
                        processed_image,
                        obb_data,
                        self.results.names,
                        self.conf_threshold,
                        line_thickness=1  # 使用细线条
                    )
            
            # 显示处理结果
            self.display_image(processed_image)
            
            # 显示处理信息
            num_detections = len(obb_data)
            contour_info = f"，提取了 {len(self.contour_points)} 个轮廓" if self.contour_points else ""
            messagebox.showinfo("掩码处理完成", f"已对 {num_detections} 个检测框进行二值化掩码处理{contour_info}")
                
        except Exception as e:
            messagebox.showerror("错误", f"掩码处理失败: {e}")
    
    def toggle_mask_display(self):
        """切换掩码显示状态"""
        self.show_mask = self.show_mask_var.get()
        print(f"掩码显示状态: {'开启' if self.show_mask else '关闭'}")
    
    
    def toggle_contours_display(self):
        """切换轮廓点显示状态"""
        self.show_contours = self.show_contours_var.get()
        print(f"轮廓点显示状态: {'开启' if self.show_contours else '关闭'}")
        
        # 如果当前有检测结果，重新处理以显示/隐藏轮廓点
        if hasattr(self, 'results') and self.results is not None:
            self.process_with_mask()
    
    def toggle_binary_roi_display(self):
        """切换二值化ROI显示状态"""
        self.show_binary_roi = self.show_binary_roi_var.get()
        print(f"二值化ROI显示状态: {'开启' if self.show_binary_roi else '关闭'}")
    
    def show_binary_rois(self):
        """显示二值化ROI"""
        if self.inference.current_image is None:
            messagebox.showwarning("警告", "请先加载图片!")
            return
        
        if self.results is None:
            messagebox.showwarning("警告", "请先进行检测!")
            return
        
        try:
            # 获取OBB数据
            obb_data = self.results.obb.data.cpu().numpy() if self.results.obb is not None else None
            
            if obb_data is None or len(obb_data) == 0:
                messagebox.showinfo("提示", "没有检测到目标，无法显示二值化ROI!")
                return
            
            # 提取并二值化ROI
            self.binary_roi_images = self.inference.extract_and_binarize_roi(
                self.inference.current_image,
                obb_data,
                self.conf_threshold
            )
            
            if not self.binary_roi_images:
                messagebox.showinfo("提示", "没有有效的ROI可以显示!")
                return
            
            # 组合显示ROI图像
            combined_roi_image = self.inference.display_binary_rois(self.binary_roi_images, max_cols=4)
            
            # 显示ROI图像
            self.display_image(combined_roi_image)
            
            # 显示ROI信息
            num_rois = len(self.binary_roi_images)
            messagebox.showinfo("二值化ROI显示", f"显示了 {num_rois} 个二值化ROI区域")
            
        except Exception as e:
            messagebox.showerror("错误", f"显示二值化ROI失败: {e}")
    
    def display_image(self, image: np.ndarray):
        """在画布上显示图片"""
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图片
        pil_image = Image.fromarray(image_rgb)
        
        # 调整图片大小以适应显示，但保持原图比例
        max_display_size = (1200, 900)  # 增大显示尺寸
        pil_image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
        
        # 转换为Tkinter格式
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # 清除画布并显示图片
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # 更新滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def prev_image(self):
        """上一张图片"""
        if not self.image_files:
            return
        
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.load_current_image()
        self.update_info()
    
    def next_image(self):
        """下一张图片"""
        if not self.image_files:
            return
        
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.load_current_image()
        self.update_info()
    
    def update_conf_threshold(self, value):
        """更新置信度阈值"""
        self.conf_threshold = float(value)
        self.conf_label.config(text=f"{self.conf_threshold:.2f}")
    
    def update_info(self):
        """更新图片信息"""
        if self.image_files:
            current_file = self.image_files[self.current_index]
            info_text = f"图片 {self.current_index + 1}/{len(self.image_files)}: {current_file.name}"
            self.info_label.config(text=info_text)
        else:
            self.info_label.config(text="")
    
    def save_result(self):
        """保存检测结果"""
        if self.results is None:
            messagebox.showwarning("警告", "没有检测结果可保存!")
            return
        
        # 选择保存路径
        save_path = filedialog.asksaveasfilename(
            title="保存检测结果",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if not save_path:
            return
        
        try:
            # 获取当前显示的图片（可能是检测结果或掩码处理结果）
            current_image = self.inference.current_image.copy()
            
            # 如果有检测结果，根据当前状态决定保存什么
            obb_data = self.results.obb.data.cpu().numpy() if self.results is not None and self.results.obb is not None else None
            
            if obb_data is not None and len(obb_data) > 0:
                # 检查是否已经进行了掩码处理（通过检查图片是否包含红色掩码）
                if self.show_mask:
                    # 进行掩码处理
                    processed_image = self.inference.process_rotated_boxes(
                        current_image,
                        obb_data,
                        self.conf_threshold,
                        True,
                        self.mask_color
                    )
                    cv2.imwrite(save_path, processed_image)
                else:
                    # 只绘制检测框
                    image_with_boxes = self.inference.draw_rotated_boxes(
                        current_image, 
                        obb_data, 
                        self.results.names,
                        self.conf_threshold,
                        line_thickness=2  # 保存时使用中等粗细
                    )
                    cv2.imwrite(save_path, image_with_boxes)
            else:
                cv2.imwrite(save_path, current_image)
            
            messagebox.showinfo("成功", f"结果已保存到: {save_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")
    
    def save_roi_images(self):
        """保存检测框内的图像"""
        if self.inference.current_image is None:
            messagebox.showwarning("警告", "请先加载图片!")
            return
        
        if self.results is None:
            messagebox.showwarning("警告", "请先进行检测!")
            return
        
        try:
            # 获取OBB数据
            obb_data = self.results.obb.data.cpu().numpy() if self.results.obb is not None else None
            
            if obb_data is None or len(obb_data) == 0:
                messagebox.showinfo("提示", "没有检测到目标，无法保存框内图像!")
                return
            
            # 提取ROI图像
            roi_images = self.inference.extract_roi_images(
                self.inference.current_image,
                obb_data,
                self.conf_threshold
            )
            
            if not roi_images:
                messagebox.showinfo("提示", "没有有效的ROI可以保存!")
                return
            
            # 选择保存文件夹
            save_folder = filedialog.askdirectory(title="选择保存ROI图像的文件夹")
            if not save_folder:
                return
            
            # 保存ROI图像
            success = self.inference.save_roi_images_to_folder(roi_images, save_folder)
            
            if success:
                messagebox.showinfo("成功", f"已保存 {len(roi_images)} 个纯净ROI图像到: {save_folder}\n\n文件说明:\n- roi_001.jpg, roi_002.jpg... : 纯净的ROI图像\n- roi_001_info.txt, roi_002_info.txt... : 检测信息文件")
            else:
                messagebox.showerror("错误", "保存ROI图像失败!")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存ROI图像失败: {e}")
    
    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主函数"""
    # 模型路径
    model_path = r"D:\Desktop\XLWD\project\work915\project1\best.pt"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    try:
        # 创建并运行GUI
        app = YOLOOBBGUI(model_path)
        app.run()
        
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
