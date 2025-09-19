#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像二值化处理脚本
支持选择图像文件，进行二值化处理并显示结果
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os

class ImageBinarizationApp:
    """图像二值化应用"""
    
    def __init__(self):
        """初始化应用"""
        self.root = tk.Tk()
        self.root.title("图像二值化处理工具")
        self.root.geometry("1000x700")
        
        self.original_image = None
        self.binary_image = None
        self.current_image = None
        
        # 二值化参数
        self.threshold_method = "OTSU"
        self.threshold_value = 127
        self.max_value = 255
        
        self.create_widgets()
    
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 文件选择按钮
        ttk.Button(control_frame, text="选择图像", 
                  command=self.load_image).pack(side=tk.LEFT, padx=(0, 10))
        
        # 二值化方法选择
        ttk.Label(control_frame, text="二值化方法:").pack(side=tk.LEFT, padx=(0, 5))
        self.method_var = tk.StringVar(value=self.threshold_method)
        method_combo = ttk.Combobox(control_frame, textvariable=self.method_var,
                                   values=["OTSU", "固定阈值", "自适应阈值", "三角法"],
                                   state="readonly", width=12)
        method_combo.pack(side=tk.LEFT, padx=(0, 10))
        method_combo.bind("<<ComboboxSelected>>", self.on_method_change)
        
        # 阈值设置（仅固定阈值时显示）
        self.threshold_frame = ttk.Frame(control_frame)
        self.threshold_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(self.threshold_frame, text="阈值:").pack(side=tk.LEFT, padx=(0, 5))
        self.threshold_var = tk.IntVar(value=self.threshold_value)
        self.threshold_scale = ttk.Scale(self.threshold_frame, from_=0, to=255,
                                        variable=self.threshold_var, orient=tk.HORIZONTAL,
                                        command=self.update_threshold)
        self.threshold_scale.pack(side=tk.LEFT, padx=(0, 5))
        
        self.threshold_label = ttk.Label(self.threshold_frame, text=str(self.threshold_value))
        self.threshold_label.pack(side=tk.LEFT, padx=(0, 5))
        
        # 处理按钮
        ttk.Button(control_frame, text="二值化处理", 
                  command=self.binarize_image).pack(side=tk.LEFT, padx=(0, 10))
        
        # 保存按钮
        ttk.Button(control_frame, text="保存结果", 
                  command=self.save_image).pack(side=tk.LEFT, padx=(0, 10))
        
        # 显示模式切换
        self.display_mode = tk.StringVar(value="original")
        ttk.Radiobutton(control_frame, text="原图", variable=self.display_mode, 
                       value="original", command=self.switch_display).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(control_frame, text="二值化", variable=self.display_mode, 
                       value="binary", command=self.switch_display).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(control_frame, text="对比", variable=self.display_mode, 
                       value="compare", command=self.switch_display).pack(side=tk.LEFT, padx=(0, 10))
        
        # 图像显示区域
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
        
        # 状态标签
        self.status_label = ttk.Label(main_frame, text="请选择图像文件")
        self.status_label.pack(pady=(10, 0))
        
        # 初始化显示
        self.update_threshold_display()
    
    def load_image(self):
        """加载图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("PNG文件", "*.png"),
                ("BMP文件", "*.bmp"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # 加载图像
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("错误", "无法加载图像文件!")
                return
            
            self.binary_image = None
            self.current_image = self.original_image.copy()
            
            # 显示图像
            self.display_image(self.current_image)
            
            # 更新状态
            height, width = self.original_image.shape[:2]
            self.status_label.config(text=f"已加载图像: {os.path.basename(file_path)} ({width}x{height})")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {e}")
    
    def on_method_change(self, event=None):
        """二值化方法变更"""
        self.threshold_method = self.method_var.get()
        self.update_threshold_display()
        print(f"二值化方法变更为: {self.threshold_method}")
    
    def update_threshold_display(self):
        """更新阈值控件显示"""
        if self.threshold_method == "固定阈值":
            self.threshold_frame.pack(side=tk.LEFT, padx=(0, 10))
        else:
            self.threshold_frame.pack_forget()
    
    def update_threshold(self, value):
        """更新阈值值"""
        self.threshold_value = int(float(value))
        self.threshold_label.config(text=str(self.threshold_value))
    
    def binarize_image(self):
        """对图像进行二值化处理"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先加载图像!")
            return
        
        try:
            # 转换为灰度图
            if len(self.original_image.shape) == 3:
                gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.original_image.copy()
            
            # 根据选择的方法进行二值化
            if self.threshold_method == "OTSU":
                _, self.binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif self.threshold_method == "固定阈值":
                _, self.binary_image = cv2.threshold(gray, self.threshold_value, 255, cv2.THRESH_BINARY)
                
            elif self.threshold_method == "自适应阈值":
                self.binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                        cv2.THRESH_BINARY, 11, 2)
                
            elif self.threshold_method == "三角法":
                _, self.binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
            
            # 转换为三通道图像用于显示
            self.binary_image = cv2.cvtColor(self.binary_image, cv2.COLOR_GRAY2BGR)
            
            # 更新显示
            self.switch_display()
            
            # 更新状态
            self.status_label.config(text=f"二值化处理完成 - 方法: {self.threshold_method}")
            
        except Exception as e:
            messagebox.showerror("错误", f"二值化处理失败: {e}")
    
    def switch_display(self):
        """切换显示模式"""
        if self.original_image is None:
            return
        
        mode = self.display_mode.get()
        
        if mode == "original":
            self.current_image = self.original_image.copy()
        elif mode == "binary" and self.binary_image is not None:
            self.current_image = self.binary_image.copy()
        elif mode == "compare" and self.binary_image is not None:
            self.current_image = self.create_comparison_image()
        else:
            return
        
        self.display_image(self.current_image)
    
    def create_comparison_image(self):
        """创建对比图像"""
        if self.original_image is None or self.binary_image is None:
            return self.original_image
        
        # 调整图像大小使其一致
        h1, w1 = self.original_image.shape[:2]
        h2, w2 = self.binary_image.shape[:2]
        
        # 使用较小的尺寸
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        
        # 调整图像大小
        orig_resized = cv2.resize(self.original_image, (target_w, target_h))
        binary_resized = cv2.resize(self.binary_image, (target_w, target_h))
        
        # 水平拼接
        comparison = np.hstack((orig_resized, binary_resized))
        
        # 添加分割线
        cv2.line(comparison, (target_w, 0), (target_w, target_h), (0, 255, 0), 2)
        
        # 添加标签
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Binary", (target_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return comparison
    
    def display_image(self, image):
        """在画布上显示图像"""
        if image is None:
            return
        
        # 转换颜色空间
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(image_rgb)
        
        # 调整图像大小以适应显示
        max_display_size = (800, 600)
        pil_image.thumbnail(max_display_size, Image.Resampling.LANCZOS)
        
        # 转换为Tkinter格式
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # 清除画布并显示图像
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # 更新滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def save_image(self):
        """保存处理结果"""
        if self.binary_image is None:
            messagebox.showwarning("警告", "没有二值化结果可保存!")
            return
        
        # 选择保存路径
        save_path = filedialog.asksaveasfilename(
            title="保存二值化结果",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG files", "*.jpg"),
                ("PNG files", "*.png"),
                ("BMP files", "*.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not save_path:
            return
        
        try:
            # 保存二值化结果
            cv2.imwrite(save_path, self.binary_image)
            messagebox.showinfo("成功", f"二值化结果已保存到: {save_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")
    
    def run(self):
        """运行应用"""
        self.root.mainloop()

def main():
    """主函数"""
    try:
        app = ImageBinarizationApp()
        app.run()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
