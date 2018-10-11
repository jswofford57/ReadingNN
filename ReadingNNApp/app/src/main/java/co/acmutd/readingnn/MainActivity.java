package co.acmutd.readingnn;

import android.graphics.Bitmap;
import android.os.PersistableBundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.Utils;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static String TAG = "ReadingNN";
    private Button clearButton, analyzeButton;
    private DrawView drawView;
    private TextView result;
    private RelativeLayout parent;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    analyzeButton.setOnClickListener(new Button.OnClickListener(){
                        @Override
                        public void onClick(View view){
                            result.setText(drawView.process());
//                            parent.setDrawingCacheEnabled(true);
//                            Bitmap b = parent.getDrawingCache(); //Bitmap.createBitmap(parent.getDrawingCache(), 0, drawView.h / 4, drawView.w, drawView.h * 3 / 4);
//                            Mat mat = new Mat(b.getWidth(), b.getHeight(), CvType.CV_8UC1);
//                            Utils.bitmapToMat(b, mat);
//
//                            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2GRAY);
//                            Imgproc.GaussianBlur(mat, mat, new Size(0, 0), 1);
//                            List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
//                            Imgproc.findContours(mat, contours, new Mat(b.getWidth(), b.getHeight(), CvType.CV_8UC1), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
//                            if(contours.size() > 0){
//                                Rect r = Imgproc.boundingRect(contours.get(0));
//                                Log.d(TAG, "Rect X: " + Integer.toString(r.x) + " Rect Y: " + Integer.toString(r.y) + " Rect Width: " + Integer.toString(r.width));
//                                drawView.drawRect(r.x, r.y, r.x + r.width, r.y + r.height);
//                            }
//                            b.recycle();
//                            Log.d("ReadingNN", "Contours: " + Integer.toString(contours.size()));
//                            Log.d("ReadingNN", contours.toString());
                        }
                    });
                    Log.i(TAG, "Set analyze listener");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        clearButton = (Button) findViewById(R.id.clear);
        analyzeButton = findViewById(R.id.analyze);
        parent = (RelativeLayout) findViewById(R.id.drawContainer);
        result = findViewById(R.id.result);
//        parent.setDrawingCacheEnabled(true);
        drawView = new DrawView(this);
        parent.addView(drawView);
        clearButton.setOnClickListener(new Button.OnClickListener(){

            @Override
            public void onClick(View view) {
                drawView.clear();
            }
        });

        Log.d(TAG, "Loading OpenCV");
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
}