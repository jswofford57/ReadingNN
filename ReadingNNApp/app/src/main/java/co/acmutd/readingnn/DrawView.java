package co.acmutd.readingnn;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by usaid on 10/6/2018.
 */

public class DrawView extends View {
    private Bitmap  mBitmap;
    private Canvas  mCanvas;
    private Path    mPath;
    private Paint   mBitmapPaint;
    private Paint   mPaint;
    private Paint   rectPaint;
    private Bitmap originalBitmap;
    private static String TAG = "ReadingNN";
    private static Size inputSize = new Size(28, 28);
    private TensorFlowInferenceInterface inferenceInterface;
    private float output[] = new float[47];
    private String outputName[] = new String[]{"output_node0"};

    public int h;
    public int w;

    public DrawView(Context c) {
        super(c);

        mPath = new Path();
        mBitmapPaint = new Paint(Paint.DITHER_FLAG);

        mPaint = new Paint();
        mPaint.setAntiAlias(true);
        mPaint.setDither(true);
        mPaint.setColor(Color.RED);
        mPaint.setStyle(Paint.Style.STROKE);
        mPaint.setStrokeJoin(Paint.Join.ROUND);
        mPaint.setStrokeCap(Paint.Cap.ROUND);
        mPaint.setStrokeWidth(15);
        rectPaint = new Paint();
        rectPaint.setColor(Color.GREEN);
        rectPaint.setStrokeWidth(10);
        rectPaint.setStyle(Paint.Style.STROKE);
        this.setDrawingCacheEnabled(true);

        inferenceInterface = new TensorFlowInferenceInterface(getContext().getAssets(), "model.pb");
        Log.d(TAG, "Draw Init");
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        mBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        originalBitmap = mBitmap.copy(mBitmap.getConfig(), true);
        mCanvas = new Canvas(mBitmap);
        Log.d(TAG, "Size Changed");
//        mCanvas.drawRect(0, h / 4, w, h / 4 + w, rectPaint);
        this.w = w;
        this.h = h;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        canvas.drawPath(mPath, mPaint);
        canvas.drawBitmap(mBitmap, 0, 0, mBitmapPaint);
        canvas.save();
        Log.d(TAG, "Draw " + Boolean.toString(mBitmap.sameAs(originalBitmap)));
    }

    protected void drawRect(int l, int t, int w, int b){
        if(mCanvas != null){
            mCanvas.drawRect(l, t, w, b, rectPaint);
        }
    }

    protected String process(){
        Mat mat = new Mat(mBitmap.getWidth(), mBitmap.getHeight(), CvType.CV_8UC1);
        Utils.bitmapToMat(mBitmap, mat);
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY);
        Imgproc.threshold(mat, mat, 0.01, 1, Imgproc.THRESH_TOZERO);
        Imgproc.GaussianBlur(mat, mat, new Size(0, 0), 1);
//        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2RGBA);
//        Utils.matToBitmap(mat, mBitmap);
//        float debugging[] = matToArray(mat);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(mat, contours, new Mat(mBitmap.getWidth(), mBitmap.getHeight(), CvType.CV_8UC1), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        if(contours.size() > 0){
            Rect r = Imgproc.boundingRect(contours.get(0));
            Log.d(TAG, "Rect X: " + Integer.toString(r.x) + " Rect Y: " + Integer.toString(r.y) + " Rect Width: " + Integer.toString(r.width));
            this.drawRect(r.x, r.y, r.x + r.width, r.y + r.height);
            Point center = new Point(r.x + r.width / 2, r.y + r.height / 2);
            int squareWidth = r.width > r.height ? r.width + 5: r.height + 5;
            this.drawRect((int)(center.x) - squareWidth / 2, (int)center.y - squareWidth / 2, (int)center.x + squareWidth / 2, (int)center.y + squareWidth / 2);
            Mat submat = mat.submat(new Rect((int)(center.x) - squareWidth / 2, (int)center.y - squareWidth / 2, squareWidth, squareWidth));
            Imgproc.resize(submat, submat, DrawView.inputSize);
            float input[] = matToArray(submat);
            //Feed mat to neural network

            inferenceInterface.feed("zero_padding2d_1_input", input, 1, 28, 28, 1);
            inferenceInterface.run(outputName);
            inferenceInterface.fetch("output_node0", output);
            float max = -1;
            int maxIndex = -1;
            for(int i = 0; i < output.length; i++){
                if(max < output[i]){
                    max = output[i];
                    maxIndex = i;
                }
            }
            return mapOutputToChar(maxIndex);
        }
        return "ERROR";
    }

    protected Bitmap getBitmap(){
        return mBitmap;
    }

    private String mapOutputToChar(int input){
        if(input >= 0 && input <= 9){
            return Integer.toString(input);
        }
        else if(input <= 35){
            return Character.toString((char) (input + 55));
        }
        else if(input == 36){
            return "a";
        }
        else if(input == 37){
            return "b";
        }
        else if(input >= 38 && input <= 42){
            return Character.toString((char) (input + 62));
        }
        else if(input == 43){
            return "n";
        }
        else if(input == 44){
            return "q";
        }
        else if(input == 45){
            return "r";
        }
        else if(input == 46){
            return "t";
        }
        else{
            return "ERROR";
        }
    }

    private float[] matToArray(Mat mat){
        float[] arr = new float[28 * 28];
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                if(mat.get(i, j)[0] > 0.0)
                    arr[i * 28 + j % 28] = 1;
                else
                    arr[i * 28 + j % 28] = 0;
            }
        }
        return arr;
    }
    private float mX, mY;
    private static final float TOUCH_TOLERANCE = 4;

    private void touch_start(float x, float y) {
        mPath.reset();
        mPath.moveTo(x, y);
        mX = x;
        mY = y;
    }
    private void touch_move(float x, float y) {
        float dx = Math.abs(x - mX);
        float dy = Math.abs(y - mY);
        if (dx >= TOUCH_TOLERANCE || dy >= TOUCH_TOLERANCE) {
            mPath.quadTo(mX, mY, (x + mX)/2, (y + mY)/2);
            mX = x;
            mY = y;
        }
    }
    private void touch_up() {
        mPath.lineTo(mX, mY);
        // commit the path to our offscreen
        mCanvas.drawPath(mPath, mPaint);
        // kill this so we don't double draw
        mPath.reset();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                touch_start(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_MOVE:
                touch_move(x, y);
                invalidate();
                break;
            case MotionEvent.ACTION_UP:
                touch_up();
                invalidate();
                break;
        }
        return true;
    }

    public void clear(){
        mBitmap.eraseColor(Color.TRANSPARENT);
        invalidate();
        System.gc();
    }
}