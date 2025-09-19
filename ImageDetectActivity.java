package com.tencent.yolo11ncnn;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.Toast;
import org.json.JSONArray;
import org.json.JSONObject;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import android.graphics.PointF;
import org.opencv.android.OpenCVLoader;

public class ImageDetectActivity extends Activity {
    private YOLO11Ncnn yolo11ncnn = new YOLO11Ncnn();
    private int current_task = 5;  // 默认使用 insulatorAndPerson 模型
    private int current_model = 0;
    private int current_cpugpu = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d("ImageDetectActivity", "onCreate start");
        
        // 初始化OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e("ImageDetectActivity", "OpenCV初始化失败");
            Toast.makeText(this, "OpenCV初始化失败", Toast.LENGTH_SHORT).show();
            finish();
            return;
        } else {
            Log.d("ImageDetectActivity", "OpenCV初始化成功");
        }
        
        // 从Intent中获取模型参数
        Intent intent = getIntent();
        current_task = intent.getIntExtra("task", 5);  // 默认使用 insulatorAndPerson 模型
        current_model = intent.getIntExtra("model", 0);
        current_cpugpu = intent.getIntExtra("cpugpu", 0);
        
        ImageView imageView = new ImageView(this);
        setContentView(imageView);

        Uri imageUri = intent.getData();
        Log.d("ImageDetectActivity", "got imageUri: " + imageUri);
        if (imageUri == null) {
            Toast.makeText(this, "未获取到图片", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            // 1. 先读取图片尺寸
            InputStream inputStream = null;
            try {
                Log.d("ImageDetectActivity", "before openInputStream for size");
                inputStream = getContentResolver().openInputStream(imageUri);
                if (inputStream == null) {
                    Toast.makeText(this, "图片流获取失败", Toast.LENGTH_SHORT).show();
                    finish();
                    return;
                }
                options.inJustDecodeBounds = true;
                BitmapFactory.decodeStream(inputStream, null, options);
                inputStream.close();
                Log.d("ImageDetectActivity", "after decodeStream for size");
            } catch (Exception e) {
                Log.e("ImageDetectActivity", "图片尺寸读取失败", e);
                Toast.makeText(this, "图片尺寸读取失败", Toast.LENGTH_SHORT).show();
                finish();
                return;
            }

            int reqWidth = 800;
            int reqHeight = 800;
            int width = options.outWidth;
            int height = options.outHeight;
            Log.d("ImageDetectActivity", "image size: " + width + "x" + height);

            // 2. 计算缩放比例
            int inSampleSize = 1;
            if (height > reqHeight || width > reqWidth) {
                final int halfHeight = height / 2;
                final int halfWidth = width / 2;
                while ((halfHeight / inSampleSize) >= reqHeight && (halfWidth / inSampleSize) >= reqWidth) {
                    inSampleSize *= 2;
                }
            }
            Log.d("ImageDetectActivity", "inSampleSize: " + inSampleSize);

            // 3. 按缩放比例加载图片
            Bitmap bitmap = null;
            try {
                Log.d("ImageDetectActivity", "before openInputStream for decode");
                options.inJustDecodeBounds = false;
                options.inSampleSize = inSampleSize;
                inputStream = getContentResolver().openInputStream(imageUri);
                if (inputStream == null) {
                    Toast.makeText(this, "图片流获取失败", Toast.LENGTH_SHORT).show();
                    finish();
                    return;
                }
                bitmap = BitmapFactory.decodeStream(inputStream, null, options);
                inputStream.close();
                Log.d("ImageDetectActivity", "after decodeStream for bitmap");
            } catch (OutOfMemoryError e) {
                Log.e("ImageDetectActivity", "图片太大，无法处理", e);
                Toast.makeText(this, "图片太大，无法处理", Toast.LENGTH_SHORT).show();
                finish();
                return;
            } catch (Exception e) {
                Log.e("ImageDetectActivity", "图片加载失败", e);
                Toast.makeText(this, "图片加载失败", Toast.LENGTH_SHORT).show();
                finish();
                return;
            }

            if (bitmap == null) {
                Log.e("ImageDetectActivity", "bitmap is null after decode");
                Toast.makeText(this, "图片加载失败", Toast.LENGTH_SHORT).show();
                finish();
                return;
            }
            if (bitmap.getConfig() != Bitmap.Config.ARGB_8888) {
                Log.d("ImageDetectActivity", "bitmap config is not ARGB_8888, converting");
                Bitmap converted = bitmap.copy(Bitmap.Config.ARGB_8888, false);
                if (converted == null) {
                    Log.e("ImageDetectActivity", "图片格式转换失败");
                    Toast.makeText(this, "图片格式转换失败", Toast.LENGTH_SHORT).show();
                    finish();
                    return;
                }
                bitmap = converted;
            }
            Log.d("ImageDetectActivity", "bitmap ready for detection");

            // 加载模型
            boolean ret_init = yolo11ncnn.loadModel(getAssets(), current_task, current_model, current_cpugpu);
            if (!ret_init) {
                Log.e("ImageDetectActivity", "yolo11ncnn loadModel failed");
                Toast.makeText(this, "模型加载失败", Toast.LENGTH_SHORT).show();
                imageView.setImageBitmap(bitmap);
                return;
            }

            // 识别 - 强制使用OBB检测
            String result = null;
            try {
                Log.d("ImageDetectActivity", "before detectImageOBB");
                result = yolo11ncnn.detectImageOBB(bitmap, current_model, current_cpugpu);
                Log.d("ImageDetectActivity", "after detectImageOBB, result: " + result);
            } catch (Exception e) {
                Log.e("ImageDetectActivity", "模型推理异常", e);
                Toast.makeText(this, "模型推理异常", Toast.LENGTH_SHORT).show();
                imageView.setImageBitmap(bitmap);
                return;
            }

            if (result == null || result.trim().isEmpty()) {
                Log.d("ImageDetectActivity", "detectImageOBB result invalid");
                Toast.makeText(this, "识别失败或未检测到目标", Toast.LENGTH_SHORT).show();
                imageView.setImageBitmap(bitmap);
                return;
            }

            // 画框 - 强制使用OBB绘制
            Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            if (mutableBitmap == null) {
                Log.e("ImageDetectActivity", "mutableBitmap is null");
                Toast.makeText(this, "图片处理失败", Toast.LENGTH_SHORT).show();
                imageView.setImageBitmap(bitmap);
                return;
            }
            Canvas canvas = new Canvas(mutableBitmap);
            Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5);
            Paint textPaint = new Paint();
            textPaint.setColor(Color.YELLOW);
            textPaint.setTextSize(40);
            try {
                Log.d("ImageDetectActivity", "before JSON parse");
                drawOBBResults(canvas, paint, textPaint, result, bitmap);  // 强制使用OBB绘制
                Log.d("ImageDetectActivity", "after JSON parse and draw");
            } catch (Exception e) {
                Log.e("ImageDetectActivity", "结果解析失败", e);
                Toast.makeText(this, "结果解析失败", Toast.LENGTH_SHORT).show();
                imageView.setImageBitmap(bitmap);
                return;
            }
            imageView.setImageBitmap(mutableBitmap);
            Log.d("ImageDetectActivity", "setImageBitmap done");
        } catch (Exception e) {
            Log.e("ImageDetectActivity", "error: ", e);
            Toast.makeText(this, "发生异常：" + e.getMessage(), Toast.LENGTH_LONG).show();
            finish();
        }
        Log.d("ImageDetectActivity", "onCreate end");
    }
    
    // 绘制普通矩形检测结果
    private void drawRectResults(Canvas canvas, Paint paint, Paint textPaint, String result) throws Exception {
        JSONArray arr = new JSONArray(result);
        for (int i = 0; i < arr.length(); i++) {
            JSONObject obj = arr.getJSONObject(i);
            int x = obj.getInt("x");
            int y = obj.getInt("y");
            int w = obj.getInt("w");
            int h = obj.getInt("h");
            String label = obj.getString("label");
            canvas.drawRect(x, y, x + w, y + h, paint);
            canvas.drawText(label, x, y - 10, textPaint);
        }
    }
    
    // 绘制OBB旋转矩形检测结果
    private void drawOBBResults(Canvas canvas, Paint paint, Paint textPaint, String result, Bitmap bitmap) throws Exception {
        JSONObject jsonResult = new JSONObject(result);
        JSONArray objects = jsonResult.getJSONArray("objects");
        
        for (int i = 0; i < objects.length(); i++) {
            JSONObject obj = objects.getJSONObject(i);
            String label = obj.getString("label");
            
            // 获取四个角点
            JSONArray vertices = obj.getJSONArray("vertices");
            if (vertices.length() >= 4) {
                // 如果是line类型，使用轮廓点连接绘制
                if ("line".equals(label)) {
                    drawLineContour(canvas, paint, textPaint, obj, bitmap);
                } else {
                    // 其他类型使用普通旋转矩形绘制
                    drawRotatedRect(canvas, paint, textPaint, vertices, label);
                }
            }
        }
    }
    
    // 绘制普通旋转矩形
    private void drawRotatedRect(Canvas canvas, Paint paint, Paint textPaint, JSONArray vertices, String label) throws Exception {
        // 绘制旋转矩形的四条边
        for (int j = 0; j < 4; j++) {
            JSONObject point1 = vertices.getJSONObject(j);
            JSONObject point2 = vertices.getJSONObject((j + 1) % 4);
            
            float x1 = (float) point1.getDouble("x");
            float y1 = (float) point1.getDouble("y");
            float x2 = (float) point2.getDouble("x");
            float y2 = (float) point2.getDouble("y");
            
            canvas.drawLine(x1, y1, x2, y2, paint);
        }
        
        // 绘制标签
        JSONObject firstPoint = vertices.getJSONObject(0);
        float x = (float) firstPoint.getDouble("x");
        float y = (float) firstPoint.getDouble("y");
        canvas.drawText(label, x, y - 10, textPaint);
    }
    
    // 绘制line类型的轮廓点连接
    private void drawLineContour(Canvas canvas, Paint paint, Paint textPaint, JSONObject obj, Bitmap bitmap) throws Exception {
        // 获取旋转矩形的基本信息
        JSONObject center = obj.getJSONObject("center");
        JSONObject size = obj.getJSONObject("size");
        double angle = obj.getDouble("angle");
        
        double centerX = center.getDouble("x");
        double centerY = center.getDouble("y");
        double width = size.getDouble("width");
        double height = size.getDouble("height");
        
        Log.d("ImageDetectActivity", "绘制line轮廓 - 中心:(" + centerX + "," + centerY + 
              ") 尺寸:(" + width + "x" + height + ") 角度:" + angle + "度");
        
        // 使用ContourExtractor提取轮廓点
        List<PointF> denseContourPoints = ContourExtractor.extractContourFromBitmap(
            bitmap, centerX, centerY, width, height, angle, 80);
        
        Log.d("ImageDetectActivity", "轮廓提取结果，点数: " + (denseContourPoints != null ? denseContourPoints.size() : 0));
        
        // 1. 先绘制旋转矩形框（蓝色）
        Paint rectPaint = new Paint();
        rectPaint.setColor(Color.BLUE);
        rectPaint.setStyle(Paint.Style.STROKE);
        rectPaint.setStrokeWidth(3);
        
        // 计算旋转矩形的四个顶点
        // 注意：YOLO OBB的角度通常是度数，需要转换为弧度
        double angleRad = Math.toRadians(angle);
        double cosAngle = Math.cos(angleRad);
        double sinAngle = Math.sin(angleRad);
        
        Log.d("ImageDetectActivity", "角度转换 - 原始角度:" + angle + "度, 弧度:" + angleRad + 
              ", cos:" + cosAngle + ", sin:" + sinAngle);
        double halfW = width / 2.0;
        double halfH = height / 2.0;
        
        // 四个顶点相对于中心的偏移
        double[][] offsets = {
            {-halfW, -halfH},
            {halfW, -halfH},
            {halfW, halfH},
            {-halfW, halfH}
        };
        
        // 应用旋转变换并平移到实际位置
        PointF[] rectCorners = new PointF[4];
        for (int i = 0; i < 4; i++) {
            double x = offsets[i][0] * cosAngle - offsets[i][1] * sinAngle + centerX;
            double y = offsets[i][0] * sinAngle + offsets[i][1] * cosAngle + centerY;
            rectCorners[i] = new PointF((float)x, (float)y);
        }
        
        // 绘制旋转矩形的四条边
        for (int i = 0; i < 4; i++) {
            PointF start = rectCorners[i];
            PointF end = rectCorners[(i + 1) % 4];
            canvas.drawLine(start.x, start.y, end.x, end.y, rectPaint);
        }
        
        // 绘制标签
        String label = obj.getString("label");
        canvas.drawText(label, (float)centerX, (float)centerY - 10, textPaint);
        
        // 如果轮廓提取成功，绘制轮廓点和轮廓线
        if (denseContourPoints != null && !denseContourPoints.isEmpty()) {
            Log.d("ImageDetectActivity", "开始绘制轮廓，总点数: " + denseContourPoints.size());
            
            // 1. 绘制轮廓点（绿色圆点）
            Paint pointPaint = new Paint();
            pointPaint.setColor(Color.GREEN);
            pointPaint.setStyle(Paint.Style.FILL);
            pointPaint.setStrokeWidth(3);
            
            for (PointF point : denseContourPoints) {
                canvas.drawCircle(point.x, point.y, 3, pointPaint);
            }
            
            // 2. 连接轮廓点（红色线条）
            Paint linePaint = new Paint();
            linePaint.setColor(Color.RED);
            linePaint.setStyle(Paint.Style.STROKE);
            linePaint.setStrokeWidth(2);
            
            for (int i = 0; i < denseContourPoints.size() - 1; i++) {
                PointF start = denseContourPoints.get(i);
                PointF end = denseContourPoints.get(i + 1);
                canvas.drawLine(start.x, start.y, end.x, end.y, linePaint);
            }
            
            // 连接最后一个点和第一个点形成闭合轮廓
            if (denseContourPoints.size() > 2) {
                PointF start = denseContourPoints.get(denseContourPoints.size() - 1);
                PointF end = denseContourPoints.get(0);
                canvas.drawLine(start.x, start.y, end.x, end.y, linePaint);
            }
        } else {
            Log.w("ImageDetectActivity", "轮廓提取失败，只绘制旋转矩形框");
        }
    }
    
    // 备用轮廓点生成方法
    private List<PointF> generateFallbackContourPoints(double centerX, double centerY, 
                                                      double width, double height, double angle, 
                                                      int samplePoints) {
        // 计算旋转矩形的四个顶点
        double cosAngle = Math.cos(angle);
        double sinAngle = Math.sin(angle);
        
        double halfW = width / 2.0;
        double halfH = height / 2.0;
        
        // 四个顶点相对于中心的偏移
        double[][] offsets = {
            {-halfW, -halfH},
            {halfW, -halfH},
            {halfW, halfH},
            {-halfW, halfH}
        };
        
        // 应用旋转变换并平移到实际位置
        List<PointF> originalPoints = new ArrayList<>();
        for (double[] offset : offsets) {
            double x = offset[0] * cosAngle - offset[1] * sinAngle + centerX;
            double y = offset[0] * sinAngle + offset[1] * cosAngle + centerY;
            originalPoints.add(new PointF((float)x, (float)y));
        }
        
        // 生成密集的轮廓点
        return generateDenseContourPoints(originalPoints, samplePoints);
    }
    
    /**
     * 生成密集的轮廓点
     */
    private List<PointF> generateDenseContourPoints(List<PointF> originalPoints, int targetCount) {
        List<PointF> densePoints = new ArrayList<>();
        
        if (originalPoints.size() < 2) {
            return originalPoints;
        }
        
        // 计算总周长
        double totalLength = 0;
        List<Double> segmentLengths = new ArrayList<>();
        
        for (int i = 0; i < originalPoints.size(); i++) {
            PointF start = originalPoints.get(i);
            PointF end = originalPoints.get((i + 1) % originalPoints.size());
            
            double length = Math.sqrt(Math.pow(end.x - start.x, 2) + Math.pow(end.y - start.y, 2));
            segmentLengths.add(length);
            totalLength += length;
        }
        
        if (totalLength == 0) {
            return originalPoints;
        }
        
        // 在每条边上均匀分布点
        double stepLength = totalLength / targetCount;
        double currentLength = 0;
        int segmentIndex = 0;
        
        for (int i = 0; i < targetCount; i++) {
            double targetLength = i * stepLength;
            
            // 找到目标长度对应的线段
            while (currentLength + segmentLengths.get(segmentIndex) < targetLength && segmentIndex < segmentLengths.size() - 1) {
                currentLength += segmentLengths.get(segmentIndex);
                segmentIndex++;
            }
            
            // 在当前线段内插值
            PointF start = originalPoints.get(segmentIndex);
            PointF end = originalPoints.get((segmentIndex + 1) % originalPoints.size());
            
            double segmentLength = segmentLengths.get(segmentIndex);
            double t = segmentLength > 0 ? (targetLength - currentLength) / segmentLength : 0;
            t = Math.max(0, Math.min(1, t)); // 限制在[0,1]范围内
            
            float x = (float)(start.x + t * (end.x - start.x));
            float y = (float)(start.y + t * (end.y - start.y));
            
            densePoints.add(new PointF(x, y));
        }
        
        return densePoints;
    }
    
} 