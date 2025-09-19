package com.tencent.yolo11ncnn;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Core;
import java.util.ArrayList;
import java.util.List;
import android.graphics.Bitmap;
import android.graphics.PointF;
import org.opencv.android.OpenCVLoader;

public class ContourExtractor {
    
    // 静态初始化块，确保OpenCV库已加载
    static {
        if (!OpenCVLoader.initDebug()) {
            System.err.println("ContourExtractor: OpenCV库加载失败！");
        } else {
            System.out.println("ContourExtractor: OpenCV库加载成功");
        }
    }
    
    /**
     * 从旋转矩形区域提取轮廓点
     * 
     * @param image 输入图像
     * @param x 旋转矩形中心点x坐标
     * @param y 旋转矩形中心点y坐标
     * @param w 旋转矩形宽度
     * @param h 旋转矩形高度
     * @param angle 旋转角度（弧度）
     * @param samplePoints 采样点数量，默认200
     * @return 轮廓点数组，形状为 (samplePoints, 2)
     */
    public static MatOfPoint extractContourFromRotatedRect(Mat image, double x, double y, 
                                                         double w, double h, double angle, 
                                                         int samplePoints) {
        try {
            System.out.println("开始轮廓提取 - 中心:(" + x + "," + y + ") 尺寸:(" + w + "x" + h + ") 角度:" + angle);
            System.out.println("图像尺寸: " + image.size().width + "x" + image.size().height);
            
            // 1. 计算旋转矩形的四个顶点
            // 注意：YOLO OBB的角度通常是度数，需要转换为弧度
            double angleRad = Math.toRadians(angle);
            double cosAngle = Math.cos(angleRad);
            double sinAngle = Math.sin(angleRad);
            
            // 计算半宽和半高
            double halfW = w / 2.0;
            double halfH = h / 2.0;
            
            // 计算四个顶点相对于中心的偏移
            MatOfPoint corners = new MatOfPoint();
            Point[] cornerPoints = {
                new Point(-halfW, -halfH),
                new Point(halfW, -halfH),
                new Point(halfW, halfH),
                new Point(-halfW, halfH)
            };
            corners.fromArray(cornerPoints);
            
            // 应用旋转变换（手动计算）
            Point[] rotatedPoints = new Point[4];
            for (int i = 0; i < 4; i++) {
                double px = cornerPoints[i].x;
                double py = cornerPoints[i].y;
                double rotatedX = px * cosAngle - py * sinAngle;
                double rotatedY = px * sinAngle + py * cosAngle;
                rotatedPoints[i] = new Point(rotatedX, rotatedY);
            }
            MatOfPoint rotatedCorners = new MatOfPoint();
            rotatedCorners.fromArray(rotatedPoints);
            
            // 平移到实际位置
            Point[] rotatedArray = rotatedCorners.toArray();
            Point[] translatedPoints = new Point[4];
            for (int i = 0; i < 4; i++) {
                translatedPoints[i] = new Point(rotatedArray[i].x + x, rotatedArray[i].y + y);
            }
            MatOfPoint rectPoints = new MatOfPoint();
            rectPoints.fromArray(translatedPoints);
            
            // 2. 在旋转矩形区域内进行二值化处理
            // 创建与原图同尺寸的掩码
            Mat binaryMask = Mat.zeros(image.size(), 0); // 0 = CV_8UC1
            
            // 创建旋转矩形的掩码
            Mat mask = Mat.zeros(image.size(), 0); // 0 = CV_8UC1
            List<MatOfPoint> maskContours = new ArrayList<>();
            maskContours.add(new MatOfPoint(rectPoints.toArray()));
            Imgproc.fillPoly(mask, maskContours, new Scalar(255, 255, 255, 0));
            
            // 在掩码区域内进行二值化处理
            Mat gray = new Mat();
            if (image.channels() == 3) {
                Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = image.clone();
            }
            
            // 使用OTSU二值化
            Mat binary = new Mat();
            Imgproc.threshold(gray, binary, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
            
            // 反转二值化结果，使较暗的区域为白色
            Core.bitwise_not(binary, binary);
            
            // 形态学操作，去除噪声
            Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel);
            Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_OPEN, kernel);
            
            // 只在旋转矩形区域内保留二值化结果
            binary.copyTo(binaryMask, mask);
            
            // 3. 对二值化掩码进行形态学操作，确保连通性
            Mat largeKernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(15, 15));
            Imgproc.morphologyEx(binaryMask, binaryMask, Imgproc.MORPH_CLOSE, largeKernel);
            
            // 4. 从二值化掩码中提取轮廓点
            // 查找轮廓
            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(binaryMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            
            System.out.println("找到轮廓数量: " + contours.size());
            
            if (contours.size() == 0) {
                System.out.println("简化函数: 没有找到轮廓");
                return null;
            }
            
            // 找到最大的轮廓
            MatOfPoint largestContour = contours.get(0);
            double maxArea = Imgproc.contourArea(largestContour);
            System.out.println("轮廓0面积: " + maxArea);
            
            for (int i = 0; i < contours.size(); i++) {
                MatOfPoint contour = contours.get(i);
                double area = Imgproc.contourArea(contour);
                System.out.println("轮廓" + i + "面积: " + area);
                if (area > maxArea) {
                    maxArea = area;
                    largestContour = contour;
                }
            }
            
            System.out.println("最大轮廓面积: " + maxArea);
            
            // 5. 均匀采样轮廓点
            // 将轮廓点展平
            Point[] contourPoints = largestContour.toArray();
            System.out.println("原始轮廓点数量: " + contourPoints.length);
            
            // 如果轮廓点数量已经小于等于采样数量，直接返回
            if (contourPoints.length <= samplePoints) {
                System.out.println("轮廓点数量小于采样数量，直接返回");
                MatOfPoint result = new MatOfPoint();
                Point[] resultPoints = new Point[contourPoints.length];
                for (int i = 0; i < contourPoints.length; i++) {
                    resultPoints[i] = new Point(contourPoints[i].x, contourPoints[i].y);
                }
                result.fromArray(resultPoints);
                return result;
            } else {
                // 计算轮廓的累积弧长
                double[] distances = new double[contourPoints.length];
                for (int i = 0; i < contourPoints.length - 1; i++) {
                    double dx = contourPoints[i + 1].x - contourPoints[i].x;
                    double dy = contourPoints[i + 1].y - contourPoints[i].y;
                    distances[i] = Math.sqrt(dx * dx + dy * dy);
                }
                
                double[] cumulativeDistances = new double[contourPoints.length + 1];
                cumulativeDistances[0] = 0;
                for (int i = 0; i < distances.length; i++) {
                    cumulativeDistances[i + 1] = cumulativeDistances[i] + distances[i];
                }
                
                // 计算总弧长
                double totalLength = cumulativeDistances[cumulativeDistances.length - 1];
                
                // 如果总弧长为0（所有点重合），返回等间隔采样
                if (totalLength == 0) {
                    double step = (double) contourPoints.length / samplePoints;
                    List<Point> sampledPoints = new ArrayList<>();
                    for (int i = 0; i < samplePoints; i++) {
                        int index = (int) (i * step);
                        if (index >= contourPoints.length) index = contourPoints.length - 1;
                        sampledPoints.add(new Point(contourPoints[index].x, contourPoints[index].y));
                    }
                    MatOfPoint result = new MatOfPoint();
                    result.fromArray(sampledPoints.toArray(new Point[0]));
                    return result;
                } else {
                    // 均匀采样：在弧长上均匀分布
                    List<Point> sampledPoints = new ArrayList<>();
                    for (int i = 0; i < samplePoints; i++) {
                        double dist = (double) i / samplePoints * totalLength;
                        
                        // 找到距离对应的索引
                        int idx = 0;
                        for (int j = 0; j < cumulativeDistances.length - 1; j++) {
                            if (dist >= cumulativeDistances[j] && dist <= cumulativeDistances[j + 1]) {
                                idx = j;
                                break;
                            }
                        }
                        idx = Math.min(idx, contourPoints.length - 2);
                        
                        // 线性插值
                        if (idx < contourPoints.length - 1) {
                            double t = (dist - cumulativeDistances[idx]) / 
                                      (cumulativeDistances[idx + 1] - cumulativeDistances[idx]);
                            double x_interp = (1 - t) * contourPoints[idx].x + t * contourPoints[idx + 1].x;
                            double y_interp = (1 - t) * contourPoints[idx].y + t * contourPoints[idx + 1].y;
                            sampledPoints.add(new Point(x_interp, y_interp));
                        } else {
                            sampledPoints.add(new Point(contourPoints[contourPoints.length - 1].x, 
                                                      contourPoints[contourPoints.length - 1].y));
                        }
                    }
                    
                    MatOfPoint result = new MatOfPoint();
                    result.fromArray(sampledPoints.toArray(new Point[0]));
                    System.out.println("采样完成，返回点数: " + sampledPoints.size());
                    return result;
                }
            }
            
        } catch (Exception e) {
            System.out.println("提取轮廓点失败: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
    
    /**
     * 重载方法，使用默认采样点数量200
     */
    public static MatOfPoint extractContourFromRotatedRect(Mat image, double x, double y, 
                                                         double w, double h, double angle) {
        return extractContourFromRotatedRect(image, x, y, w, h, angle, 200);
    }
    
    /**
     * 从Android Bitmap中提取轮廓点（专门用于line类型检测结果）
     * 
     * @param bitmap Android Bitmap图像
     * @param centerX 旋转矩形中心点x坐标
     * @param centerY 旋转矩形中心点y坐标
     * @param width 旋转矩形宽度
     * @param height 旋转矩形高度
     * @param angle 旋转角度（弧度）
     * @param samplePoints 采样点数量，默认100
     * @return 轮廓点列表
     */
    public static List<PointF> extractContourFromBitmap(Bitmap bitmap, double centerX, double centerY, 
                                                       double width, double height, double angle, 
                                                       int samplePoints) {
        try {
            // 将Android Bitmap转换为OpenCV Mat
            Mat image = bitmapToMat(bitmap);
            if (image == null || image.empty()) {
                System.out.println("Bitmap转Mat失败，返回空");
                return null;
            }
            
            // 使用现有的轮廓提取方法
            MatOfPoint contourMat = extractContourFromRotatedRect(image, centerX, centerY, width, height, angle, samplePoints);
            
            if (contourMat == null || contourMat.empty()) {
                System.out.println("轮廓提取失败，返回空");
                return null;
            }
            
            // 将Mat转换为PointF列表
            List<PointF> contourPoints = new ArrayList<>();
            Point[] points = contourMat.toArray();
            
            for (Point point : points) {
                contourPoints.add(new PointF((float)point.x, (float)point.y));
            }
            
            return contourPoints;
            
        } catch (Exception e) {
            System.out.println("从Bitmap提取轮廓点失败: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * 重载方法，使用默认采样点数量100
     */
    public static List<PointF> extractContourFromBitmap(Bitmap bitmap, double centerX, double centerY, 
                                                       double width, double height, double angle) {
        return extractContourFromBitmap(bitmap, centerX, centerY, width, height, angle, 200);
    }
    
    /**
     * 将Android Bitmap转换为OpenCV Mat
     */
    private static Mat bitmapToMat(Bitmap bitmap) {
        try {
            int width = bitmap.getWidth();
            int height = bitmap.getHeight();
            
            System.out.println("Bitmap转Mat - 尺寸: " + width + "x" + height);
            
            // 创建Mat - 使用正确的类型常量
            Mat mat = new Mat(height, width, 16); // 16 = CV_8UC3 (3通道8位无符号整数)
            
            // 获取像素数据
            int[] pixels = new int[width * height];
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
            
            // 转换ARGB到BGR
            byte[] data = new byte[width * height * 3];
            for (int i = 0; i < pixels.length; i++) {
                int pixel = pixels[i];
                int a = (pixel >> 24) & 0xFF; // Alpha通道
                int r = (pixel >> 16) & 0xFF;
                int g = (pixel >> 8) & 0xFF;
                int b = pixel & 0xFF;
                
                // 如果Alpha通道为0，设置为白色背景
                if (a == 0) {
                    r = g = b = 255;
                }
                
                data[i * 3] = (byte) b;     // B
                data[i * 3 + 1] = (byte) g; // G
                data[i * 3 + 2] = (byte) r; // R
            }
            
            // 将数据复制到Mat
            mat.put(0, 0, data);
            
            System.out.println("Bitmap转Mat成功");
            return mat;
        } catch (Exception e) {
            System.out.println("Bitmap转Mat失败: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
}
