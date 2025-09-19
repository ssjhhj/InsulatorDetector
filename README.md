# Java 轮廓提取器

这是Python `extract_contour_from_rotated_rect` 函数的Java实现版本。

## 功能

从旋转矩形区域提取轮廓点，支持：
- 旋转矩形顶点计算
- 区域二值化处理
- 轮廓检测与合并
- 基于弧长的均匀采样

## 依赖

- Java 8+
- OpenCV 4.x
- Maven

## 安装

1. 确保已安装Java 8+和Maven
2. 下载OpenCV并配置环境变量
3. 运行以下命令：

```bash
mvn clean compile
mvn exec:java
```

## 使用方法

### 基本用法

```java
// 加载OpenCV库
System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

// 加载图像
Mat image = Imgcodecs.imread("your_image.jpg");

// 提取轮廓点
MatOfPoint2f contourPoints = ContourExtractor.extractContourFromRotatedRect(
    image, x, y, w, h, angle, samplePoints
);

// 使用轮廓点
if (contourPoints != null) {
    Point[] points = contourPoints.toArray();
    // 处理轮廓点...
}
```

### 参数说明

- `image`: 输入图像 (Mat)
- `x, y`: 旋转矩形中心点坐标
- `w, h`: 旋转矩形的宽度和高度
- `angle`: 旋转角度（弧度）
- `samplePoints`: 采样点数量，默认200

### 返回值

- `MatOfPoint2f`: 轮廓点数组，包含指定数量的均匀分布点
- `null`: 如果提取失败

## 算法流程

1. **计算旋转矩形顶点**：使用旋转矩阵计算四个顶点
2. **区域二值化**：在旋转矩形区域内进行OTSU二值化
3. **形态学操作**：使用MORPH_CLOSE连接分离区域
4. **轮廓检测**：查找所有外部轮廓
5. **轮廓合并**：如果有多个轮廓，合并并计算凸包
6. **均匀采样**：基于弧长进行均匀采样

## 示例

```java
public static void main(String[] args) {
    System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    
    Mat image = Imgcodecs.imread("test.jpg");
    MatOfPoint2f points = ContourExtractor.extractContourFromRotatedRect(
        image, 328.6, 213.9, 469.6, 20.8, 3.116, 200
    );
    
    if (points != null) {
        System.out.println("提取到 " + points.rows() + " 个轮廓点");
    }
}
```

## 注意事项

1. 确保OpenCV库正确安装和配置
2. 输入图像路径必须存在
3. 旋转角度使用弧度制
4. 采样点数量建议设置为200以获得最佳效果

## 与Python版本的对应关系

| Python | Java |
|--------|------|
| `numpy.array` | `Mat` |
| `cv2.findContours` | `Imgproc.findContours` |
| `cv2.morphologyEx` | `Imgproc.morphologyEx` |
| `cv2.convexHull` | `Imgproc.convexHull` |
| `np.linspace` | 手动实现 |
| `np.searchsorted` | 手动实现 |
