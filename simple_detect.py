#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的YOLO-OBB检测脚本
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# 添加ultralytics到路径
sys.path.append(str(Path(__file__).parent / "ultralytics"))

try:
    from ultralytics import YOLO
except ImportError as e:
    print(f"导入ultralytics失败: {e}")
    print("请确保已安装ultralytics: pip install ultralytics")
    sys.exit(1)

def main():
    """主函数"""
    # 模型路径
    model_path = r"D:\Desktop\XLWD\project\work915\project1\best.pt"
    
    # 图片路径
    image_path = r"D:\Desktop\XLWD\project\work915\project1\line2.jpg"
    
    # 检查文件是否存在
    if not Path(model_path).exists():
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    if not Path(image_path).exists():
        print(f"错误: 图片文件不存在: {image_path}")
        return
    
    try:
        # 加载模型
        print("正在加载模型...")
        model = YOLO(model_path)
        print("模型加载成功!")
        
        # 加载图片
        print("正在加载图片...")
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法加载图片: {image_path}")
            return
        print(f"图片加载成功! 尺寸: {image.shape}")
        
        # 进行检测
        print("正在进行检测...")
        results = model(image, conf=0.25, verbose=False)
        result = results[0]
        
        # 获取检测结果
        if result.obb is not None:
            obb_data = result.obb.data.cpu().numpy()
            print(f"检测到 {len(obb_data)} 个目标")
            
            # 绘制检测框
            image_with_boxes = image.copy()
            
            for i, detection in enumerate(obb_data):
                if len(detection) < 7:
                    continue
                
                x, y, w, h, angle, conf, cls = detection[:7]
                
                # 计算旋转矩形的四个顶点
                rect_points = get_rotated_rect_points(x, y, w, h, angle)
                
                # 绘制旋转矩形
                cv2.polylines(image_with_boxes, [rect_points], True, (0, 255, 0), 2)
                
                # 绘制标签
                label = f"Class {int(cls)}: {conf:.2f}"
                cv2.putText(image_with_boxes, label, (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                print(f"目标 {i+1}: 类别={int(cls)}, 置信度={conf:.3f}, 位置=({x:.1f}, {y:.1f}), 尺寸=({w:.1f}, {h:.1f}), 角度={angle:.3f}")
        else:
            print("未检测到任何目标")
            image_with_boxes = image
        
        # 显示结果
        print("显示检测结果...")
        cv2.imshow("YOLO-OBB Detection", image_with_boxes)
        
        # 等待按键
        print("按任意键关闭检测结果...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 如果有检测结果，显示框内图片的二值化结果
        if result.obb is not None and len(obb_data) > 0:
            
            print("显示二值化掩码轮廓...")
            show_binary_mask_contours(image, obb_data)
            
            print("显示轮廓点...")
            show_contour_points(image, obb_data)
            
            # 演示简化函数的使用
            print("\n=== 使用简化函数提取轮廓点 ===")
            for i, detection in enumerate(obb_data):
                if len(detection) < 7:
                    continue
                x, y, w, h, angle, conf, cls = detection[:7]
                print(f"目标 {i+1}: 参数=({x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}, {angle:.3f})")
                
                # 使用简化函数提取轮廓点
                contour_points = extract_contour_from_rotated_rect(image, x, y, w, h, angle, sample_points=200)
                if contour_points is not None:
                    print(f"  提取到 {len(contour_points)} 个轮廓点")
                    
                    # 显示轮廓点
                    contour_image = image.copy()
                    
                    # 绘制轮廓点
                    for point in contour_points:
                        cv2.circle(contour_image, tuple(point), 2, (0, 255, 0), -1)
                    
                    # 连接轮廓点
                    for j in range(len(contour_points)):
                        start_point = tuple(contour_points[j])
                        end_point = tuple(contour_points[(j + 1) % len(contour_points)])
                        cv2.line(contour_image, start_point, end_point, (0, 255, 0), 1)
                    
                    # 绘制旋转矩形
                    rect_points = get_rotated_rect_points(x, y, w, h, angle)
                    cv2.polylines(contour_image, [rect_points], True, (255, 0, 0), 1)
                    
                    # 显示结果
                    cv2.imshow(f"目标 {i+1} 轮廓点", contour_image)
                    print(f"  按任意键关闭目标 {i+1} 的轮廓点显示...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                else:
                    print("  轮廓点提取失败")
        
    except Exception as e:
        print(f"检测失败: {e}")
        import traceback
        traceback.print_exc()

def show_binary_mask_contours(image, obb_data):
    """显示二值化掩码轮廓 - 整体掩码版本"""
    try:
        # 创建原图副本用于显示掩码
        mask_image = image.copy()
        
        # 创建整体掩码
        combined_mask = create_combined_binary_mask(image, obb_data)
        
        if combined_mask is not None:
            # 将整体二值化结果映射到掩码区域
            mask_3channel = np.zeros_like(image)
            mask_3channel[combined_mask > 0] = (0, 0, 255)  # 红色掩码
            
            # 将处理结果叠加到原图
            mask_image = cv2.addWeighted(mask_image, 0.7, mask_3channel, 0.3, 0)
            
            # 绘制所有检测框
            for i, detection in enumerate(obb_data):
                if len(detection) < 7:
                    continue
                x, y, w, h, angle, conf, cls = detection[:7]
                rect_points = get_rotated_rect_points(x, y, w, h, angle)
                cv2.polylines(mask_image, [rect_points], True, (0, 255, 0), 1)
        
        # 显示掩码结果
        cv2.imshow("Binary Mask Contours", mask_image)
        print("按任意键关闭二值化掩码轮廓...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"显示二值化掩码轮廓失败: {e}")

def binarize_rotated_region(image, rect_points):
    """直接在旋转矩形区域内进行二值化处理"""
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

def show_contour_points(image, obb_data):
    """显示轮廓点 - 整体掩码版本"""
    try:
        # 创建原图副本用于显示轮廓点
        contour_image = image.copy()
        
        # 创建整体掩码
        combined_mask = create_combined_binary_mask(image, obb_data)
        
        if combined_mask is not None:
            # 从整体掩码中提取轮廓点
            contour_points = extract_contour_points_from_mask(combined_mask, 200)
            
            if contour_points is not None and len(contour_points) > 0:
                # 绘制轮廓点
                for point in contour_points:
                    cv2.circle(contour_image, tuple(point), 2, (0, 255, 0), -1)
                
                # 连接轮廓点
                for j in range(len(contour_points)):
                    start_point = tuple(contour_points[j])
                    end_point = tuple(contour_points[(j + 1) % len(contour_points)])
                    cv2.line(contour_image, start_point, end_point, (0, 255, 0), 1)
                
                print(f"整体掩码: 提取了 {len(contour_points)} 个轮廓点")
            
            # 绘制所有检测框
            for i, detection in enumerate(obb_data):
                if len(detection) < 7:
                    continue
                x, y, w, h, angle, conf, cls = detection[:7]
                rect_points = get_rotated_rect_points(x, y, w, h, angle)
                cv2.polylines(contour_image, [rect_points], True, (255, 0, 0), 1)
        
        # 显示轮廓点结果
        cv2.imshow("Contour Points", contour_image)
        print("按任意键关闭轮廓点显示...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"显示轮廓点失败: {e}")

def create_combined_binary_mask(image, obb_data):
    """创建所有检测框的整体二值化掩码"""
    try:
        # 创建整体掩码
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for i, detection in enumerate(obb_data):
            if len(detection) < 7:
                continue
            
            x, y, w, h, angle, conf, cls = detection[:7]
            
            # 计算旋转矩形的四个顶点
            rect_points = get_rotated_rect_points(x, y, w, h, angle)
            
            # 在旋转矩形区域内进行二值化处理
            binary_mask = binarize_rotated_region(image, rect_points)
            
            # 将当前掩码合并到整体掩码中
            combined_mask = cv2.bitwise_or(combined_mask, binary_mask)
            
            print(f"处理目标 {i+1} 并合并到整体掩码")
        
        # 对整体掩码进行形态学操作，确保连通性
        print(f"合并前轮廓数: {len(cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])}")
        
        # 使用更大的核进行CLOSE操作，连接分离的区域
        kernel = np.ones((15, 15), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        print(f"CLOSE操作后有效像素数: {np.sum(combined_mask > 0)}")
        print(f"CLOSE操作后轮廓数: {len(cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])}")
        
        # 不进行OPEN操作，避免断开连接
        # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        print(f"最终有效像素数: {np.sum(combined_mask > 0)}")
        print(f"最终轮廓数: {len(cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])}")
        
        return combined_mask
        
    except Exception as e:
        print(f"创建整体掩码失败: {e}")
        return None

def extract_contour_points_from_mask(binary_mask, sample_points=200):
    """直接从二值化掩码中提取轮廓点 - 合并版本"""
    try:
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        print(f"检测到 {len(contours)} 个轮廓")
        
        if not contours:
            print("没有找到轮廓")
            return None
        
        # 打印所有轮廓的面积和位置
        areas = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            areas.append(area)
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)
            print(f"轮廓 {i+1}: 面积={area:.0f}, 位置=({x},{y},{w},{h})")
        
        # 如果仍然有多个轮廓，合并所有轮廓
        if len(contours) > 1:
            print(f"仍有 {len(contours)} 个轮廓，尝试合并所有轮廓")
            # 合并所有轮廓点
            all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
            # 计算凸包
            largest_contour = cv2.convexHull(all_points.astype(np.int32))
            print(f"合并后轮廓面积: {cv2.contourArea(largest_contour):.0f}")
        else:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            print(f"选择最大轮廓，面积: {largest_area:.0f}")
        
        # 检查轮廓面积是否足够大
        contour_area = cv2.contourArea(largest_contour)
        if contour_area < 1000:  # 面积阈值
            print(f"轮廓面积太小 ({contour_area:.0f})，尝试使用所有轮廓点")
            # 如果轮廓面积太小，使用所有轮廓点
            all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
            largest_contour = cv2.convexHull(all_points.astype(np.int32))
            print(f"使用所有轮廓点后面积: {cv2.contourArea(largest_contour):.0f}")
        
        # 计算最大轮廓的边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        print(f"最大轮廓边界框: ({x},{y},{w},{h})")
        
        # 直接对原始轮廓进行等间距采样
        contour_points = sample_contour_points(largest_contour, sample_points)
        
        if contour_points is not None:
            print(f"采样得到 {len(contour_points)} 个轮廓点")
            # 计算采样点的边界
            min_x, min_y = np.min(contour_points, axis=0)
            max_x, max_y = np.max(contour_points, axis=0)
            print(f"采样点范围: ({min_x},{min_y}) 到 ({max_x},{max_y})")
        
        return contour_points
        
    except Exception as e:
        print(f"从掩码提取轮廓点失败: {e}")
        return None


def extract_contour_from_rotated_rect(image, x, y, w, h, angle, sample_points=200):
    """
    简化函数：从旋转矩形区域提取轮廓点 - 完全展开版本
    
    参数:
        image: 输入图像
        x, y, w, h, angle: 旋转矩形参数 (中心点坐标、宽度、高度、角度)
        sample_points: 采样点数量，默认200
    
    返回:
        contour_points: 轮廓点数组，形状为 (sample_points, 2)
    """
    try:
        # 1. 计算旋转矩形的四个顶点
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        # 计算半宽和半高
        half_w = w / 2.0
        half_h = h / 2.0
        
        # 计算四个顶点相对于中心的偏移
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ])
        
        # 应用旋转变换
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        rotated_corners = np.dot(corners, rotation_matrix.T)
        
        # 平移到实际位置
        rect_points = rotated_corners + np.array([x, y])
        rect_points = rect_points.astype(np.int32)
        
        # 2. 在旋转矩形区域内进行二值化处理
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
        
        # 使用自适应阈值二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 反转二值化结果，使较暗的区域为白色
        binary = cv2.bitwise_not(binary)
                # 显示二值化结果
        cv2.imshow("反转二值化结果", binary)
        print("按任意键关闭二值化结果显示...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 形态学操作，去除噪声
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 只在旋转矩形区域内保留二值化结果
        binary_mask[mask > 0] = binary[mask > 0]
        
        # 3. 对二值化掩码进行形态学操作，确保连通性
        kernel = np.ones((15, 15), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 4. 从二值化掩码中提取轮廓点
        # 查找轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("简化函数: 没有找到轮廓")
            return None
        
        # 如果仍然有多个轮廓，合并所有轮廓
        if len(contours) > 1:
            # 合并所有轮廓点
            all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])
            # 计算凸包
            largest_contour = cv2.convexHull(all_points.astype(np.int32))
        else:
            # 找到最大的轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
        
        # 5. 均匀采样轮廓点
        # 将轮廓点展平
        contour_flat = largest_contour.reshape(-1, 2)
        
        # 如果轮廓点数量已经小于等于采样数量，直接返回
        if len(contour_flat) <= sample_points:
            contour_points = contour_flat
        else:
            # 计算轮廓的累积弧长
            distances = np.sqrt(np.sum(np.diff(contour_flat, axis=0)**2, axis=1))
            cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
            
            # 计算总弧长
            total_length = cumulative_distances[-1]
            
            # 如果总弧长为0（所有点重合），返回等间隔采样
            if total_length == 0:
                step = len(contour_flat) / sample_points
                indices = np.arange(0, len(contour_flat), step).astype(int)
                indices = indices[:sample_points]
                contour_points = contour_flat[indices].astype(np.int32)
            else:
                # 均匀采样：在弧长上均匀分布
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
                
                contour_points = sampled_points.astype(np.int32)
        
        return contour_points
        
    except Exception as e:
        print(f"提取轮廓点失败: {e}")
        return None

def demo_extract_contour():
    """
    使用示例：演示如何使用简化函数提取轮廓点
    """
    try:
        # 加载图像
        image_path = r"D:\Desktop\XLWD\dataset\dataset-pad\InsulatorPersonAndLine_rol_detect\line\images_crop\line_0050.jpg"
        image = cv2.imread(image_path)
        
        if image is None:
            print("无法加载图像")
            return
        
        # 示例参数（您可以根据实际检测结果调整）
        x, y, w, h, angle = 625.0, 411.7, 970.4, 25.1, 3.141
        
        # 提取轮廓点
        contour_points = extract_contour_from_rotated_rect(image, x, y, w, h, angle, sample_points=200)
        
        if contour_points is not None:
            print(f"成功提取 {len(contour_points)} 个轮廓点")
            
            # 可视化结果
            result_image = image.copy()
            
            # 绘制轮廓点
            for point in contour_points:
                cv2.circle(result_image, tuple(point), 2, (0, 255, 0), -1)
            
            # 连接轮廓点
            for i in range(len(contour_points)):
                start_point = tuple(contour_points[i])
                end_point = tuple(contour_points[(i + 1) % len(contour_points)])
                cv2.line(result_image, start_point, end_point, (0, 255, 0), 1)
            
            # 绘制旋转矩形
            rect_points = get_rotated_rect_points(x, y, w, h, angle)
            cv2.polylines(result_image, [rect_points], True, (255, 0, 0), 2)
            
            # 显示结果
            cv2.imshow("Contour Points Demo", result_image)
            print("按任意键关闭...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            print("轮廓点提取失败")
            
    except Exception as e:
        print(f"演示失败: {e}")

def sample_contour_points(contour, sample_points):
    """均匀采样轮廓点 - 基于弧长的真正均匀采样"""
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
        
        # 如果总弧长为0（所有点重合），返回等间隔采样
        if total_length == 0:
            step = len(contour_flat) / sample_points
            indices = np.arange(0, len(contour_flat), step).astype(int)
            indices = indices[:sample_points]
            return contour_flat[indices].astype(np.int32)
        
        # 均匀采样：在弧长上均匀分布
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
        print(f"均匀采样轮廓点失败: {e}")
        # 如果出错，返回等间隔采样作为备选
        contour_flat = contour.reshape(-1, 2)
        if len(contour_flat) > sample_points:
            step = len(contour_flat) / sample_points
            indices = np.arange(0, len(contour_flat), step).astype(int)
            indices = indices[:sample_points]
            return contour_flat[indices].astype(np.int32)
        else:
            return contour_flat

def extract_rotated_roi(image, rect_points):
    """提取旋转矩形区域"""
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

def binarize_roi(roi):
    """对ROI进行二值化处理"""
    try:
        # 转换为灰度图
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
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

def get_rotated_rect_points(x, y, w, h, angle):
    """计算旋转矩形的四个顶点"""
    import math
    
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

if __name__ == "__main__":
    main()
