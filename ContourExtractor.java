import org.bytedeco.javacpp.*;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import java.util.ArrayList;
import java.util.List;

public class ContourExtractor {
    
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
    public static Mat extractContourFromRotatedRect(Mat image, double x, double y, 
                                                   double w, double h, double angle, 
                                                   int samplePoints) {
        try {
            // 1. 计算旋转矩形的四个顶点
            double cosAngle = Math.cos(angle);
            double sinAngle = Math.sin(angle);
            
            // 计算半宽和半高
            double halfW = w / 2.0;
            double halfH = h / 2.0;
            
            // 计算四个顶点相对于中心的偏移
            Mat corners = new Mat(4, 1, CV_32FC2);
            corners.put(0, 0, -halfW, -halfH);
            corners.put(1, 0, halfW, -halfH);
            corners.put(2, 0, halfW, halfH);
            corners.put(3, 0, -halfW, halfH);
            
            // 应用旋转变换
            Mat rotationMatrix = new Mat(2, 2, CV_64F);
            rotationMatrix.put(0, 0, cosAngle, -sinAngle);
            rotationMatrix.put(1, 0, sinAngle, cosAngle);
            
            Mat rotatedCorners = new Mat();
            opencv_core.transform(corners, rotatedCorners, rotationMatrix);
            
            // 平移到实际位置
            Mat rectPoints = new Mat();
            Point2f[] points = new Point2f[4];
            rotatedCorners.get(0, 0, points);
            Point2f[] translatedPoints = new Point2f[4];
            for (int i = 0; i < 4; i++) {
                translatedPoints[i] = new Point2f(points[i].x() + (float)x, points[i].y() + (float)y);
            }
            rectPoints.put(0, 0, translatedPoints);
            
            // 2. 在旋转矩形区域内进行二值化处理
            // 创建与原图同尺寸的掩码
            Mat binaryMask = Mat.zeros(image.size(), CV_8UC1).asMat();
            
            // 创建旋转矩形的掩码
            Mat mask = Mat.zeros(image.size(), CV_8UC1).asMat();
            MatVector maskContours = new MatVector(1);
            maskContours.put(0, rectPoints);
            opencv_imgproc.fillPoly(mask, maskContours, new Scalar(255, 255, 255, 0));
            
            // 在掩码区域内进行二值化处理
            Mat gray = new Mat();
            if (image.channels() == 3) {
                opencv_imgproc.cvtColor(image, gray, opencv_imgproc.COLOR_BGR2GRAY);
            } else {
                gray = image.clone();
            }
            
            // 使用OTSU二值化
            Mat binary = new Mat();
            opencv_imgproc.threshold(gray, binary, 0, 255, opencv_imgproc.THRESH_BINARY + opencv_imgproc.THRESH_OTSU);
            
            // 反转二值化结果，使较暗的区域为白色
            opencv_core.bitwise_not(binary, binary);
            
            // 形态学操作，去除噪声
            Mat kernel = opencv_imgproc.getStructuringElement(opencv_imgproc.MORPH_RECT, new Size(3, 3));
            opencv_imgproc.morphologyEx(binary, binary, opencv_imgproc.MORPH_CLOSE, kernel);
            opencv_imgproc.morphologyEx(binary, binary, opencv_imgproc.MORPH_OPEN, kernel);
            
            // 只在旋转矩形区域内保留二值化结果
            binary.copyTo(binaryMask, mask);
            
            // 3. 对二值化掩码进行形态学操作，确保连通性
            Mat largeKernel = opencv_imgproc.getStructuringElement(opencv_imgproc.MORPH_RECT, new Size(15, 15));
            opencv_imgproc.morphologyEx(binaryMask, binaryMask, opencv_imgproc.MORPH_CLOSE, largeKernel);
            
            // 4. 从二值化掩码中提取轮廓点
            // 查找轮廓
            MatVector contours = new MatVector();
            Mat hierarchy = new Mat();
            opencv_imgproc.findContours(binaryMask, contours, hierarchy, opencv_imgproc.RETR_EXTERNAL, opencv_imgproc.CHAIN_APPROX_SIMPLE);
            
            if (contours.size() == 0) {
                System.out.println("简化函数: 没有找到轮廓");
                return null;
            }
            
            // 找到最大的轮廓
            Mat largestContour = contours.get(0);
            double maxArea = opencv_imgproc.contourArea(largestContour);
            for (int i = 0; i < contours.size(); i++) {
                Mat contour = contours.get(i);
                double area = opencv_imgproc.contourArea(contour);
                if (area > maxArea) {
                    maxArea = area;
                    largestContour = contour;
                }
            }
            
            // 5. 均匀采样轮廓点
            // 将轮廓点展平
            Point2f[] contourPoints = new Point2f[largestContour.rows()];
            largestContour.get(0, 0, contourPoints);
            
            // 如果轮廓点数量已经小于等于采样数量，直接返回
            if (contourPoints.length <= samplePoints) {
                Mat result = new Mat(contourPoints.length, 1, CV_32FC2);
                for (int i = 0; i < contourPoints.length; i++) {
                    result.put(i, 0, contourPoints[i].x(), contourPoints[i].y());
                }
                return result;
            } else {
                // 计算轮廓的累积弧长
                double[] distances = new double[contourPoints.length];
                for (int i = 0; i < contourPoints.length - 1; i++) {
                    double dx = contourPoints[i + 1].x() - contourPoints[i].x();
                    double dy = contourPoints[i + 1].y() - contourPoints[i].y();
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
                    List<Point2f> sampledPoints = new ArrayList<>();
                    for (int i = 0; i < samplePoints; i++) {
                        int index = (int) (i * step);
                        if (index >= contourPoints.length) index = contourPoints.length - 1;
                        sampledPoints.add(contourPoints[index]);
                    }
                    Mat result = new Mat(sampledPoints.size(), 1, CV_32FC2);
                    for (int i = 0; i < sampledPoints.size(); i++) {
                        result.put(i, 0, sampledPoints.get(i).x(), sampledPoints.get(i).y());
                    }
                    return result;
                } else {
                    // 均匀采样：在弧长上均匀分布
                    List<Point2f> sampledPoints = new ArrayList<>();
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
                            double x_interp = (1 - t) * contourPoints[idx].x() + t * contourPoints[idx + 1].x();
                            double y_interp = (1 - t) * contourPoints[idx].y() + t * contourPoints[idx + 1].y();
                            sampledPoints.add(new Point2f((float)x_interp, (float)y_interp));
                        } else {
                            sampledPoints.add(contourPoints[contourPoints.length - 1]);
                        }
                    }
                    
                    Mat result = new Mat(sampledPoints.size(), 1, CV_32FC2);
                    for (int i = 0; i < sampledPoints.size(); i++) {
                        result.put(i, 0, sampledPoints.get(i).x(), sampledPoints.get(i).y());
                    }
                    return result;
                }
            }
            
        } catch (Exception e) {
            System.out.println("提取轮廓点失败: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * 重载方法，使用默认采样点数量200
     */
    public static Mat extractContourFromRotatedRect(Mat image, double x, double y, 
                                                   double w, double h, double angle) {
        return extractContourFromRotatedRect(image, x, y, w, h, angle, 200);
    }
    
    /**
     * 使用示例
     */
    public static void main(String[] args) {
        // 加载图像
        Mat image = opencv_imgcodecs.imread("your_image.jpg");
        if (image.empty()) {
            System.out.println("无法加载图像");
            return;
        }
        
        // 示例参数
        double x = 328.6, y = 213.9, w = 469.6, h = 20.8, angle = 3.116;
        
        // 提取轮廓点
        Mat contourPoints = extractContourFromRotatedRect(image, x, y, w, h, angle, 200);
        
        if (contourPoints != null) {
            System.out.println("成功提取 " + contourPoints.rows() + " 个轮廓点");
            
            // 可视化结果
            Mat resultImage = image.clone();
            Point2f[] points = new Point2f[contourPoints.rows()];
            contourPoints.get(0, 0, points);
            
            // 绘制轮廓点
            for (Point2f point : points) {
                opencv_imgproc.circle(resultImage, new Point((int)point.x(), (int)point.y()), 2, new Scalar(0, 255, 0, 0), -1);
            }
            
            // 连接轮廓点
            for (int i = 0; i < points.length; i++) {
                Point2f startPoint = points[i];
                Point2f endPoint = points[(i + 1) % points.length];
                opencv_imgproc.line(resultImage, new Point((int)startPoint.x(), (int)startPoint.y()), 
                                   new Point((int)endPoint.x(), (int)endPoint.y()), new Scalar(0, 255, 0, 0), 1);
            }
            
            // 保存结果
            opencv_imgcodecs.imwrite("contour_result.jpg", resultImage);
            System.out.println("结果已保存为 contour_result.jpg");
            
        } else {
            System.out.println("轮廓点提取失败");
        }
    }
}
