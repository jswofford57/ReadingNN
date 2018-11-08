package co.acmutd.readingnn;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;

import org.opencv.android.JavaCameraView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.util.ArrayList;
import java.util.List;

import static co.acmutd.readingnn.DrawView.matToArray;

// OpenCV Classes

public class CameraActivity extends AppCompatActivity implements CvCameraViewListener2 {

    // Used for logging success or failure messages
    private static final String TAG = "OCVSample::Activity";

    // Loads camera view of OpenCV for us to use. This lets us see using OpenCV
    private CameraBridgeViewBase mOpenCvCameraView;

    // Used in Camera selection from menu (when implemented)
    private boolean              mIsJavaCamera = true;
    private MenuItem             mItemSwitchCamera = null;

    // These variables are used (at the moment) to fix camera orientation from 270degree to 0degree
    Mat mRgba;
    Mat mRgbaF;
    Mat mRgbaT;
    Mat erodeKernel;
    Point erodeAnchor;
    List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    VisionPipeline pipeline;

    TextView prediction;
    private TensorFlowInferenceInterface inferenceInterface;
    private float output[] = new float[47];
    private String outputName[] = new String[]{"output_node0"};

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    public CameraActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE); getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.show_camera);

        mOpenCvCameraView = (JavaCameraView) findViewById(R.id.show_camera_activity_java_surface_view);

        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);

        mOpenCvCameraView.setCvCameraViewListener(this);

        pipeline = new VisionPipeline();
        erodeKernel = new Mat();
        erodeAnchor = new Point(-1, -1);

        prediction = findViewById(R.id.prediction);
        inferenceInterface = new TensorFlowInferenceInterface(getApplicationContext().getAssets(), "model.pb");
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
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

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {

        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaF = new Mat(height, width, CvType.CV_8UC4);
        mRgbaT = new Mat(width, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        // TODO Auto-generated method stub
        mRgba = inputFrame.rgba();
//        Core.transpose(mRgba, mRgba);
        contours.clear();
//        pipeline.process(mRgba);
        Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2BGR);
        Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_BGR2HSV);
        Core.inRange(mRgba, new Scalar(43, 100, 84), new Scalar(119, 255, 232), mRgba);
        Imgproc.erode(mRgba, mRgba, erodeKernel, erodeAnchor, 7);
        Imgproc.findContours(mRgba, contours, new Mat(mRgba.width(), mRgba.height(), CvType.CV_8UC1), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        if(contours.size() > 0){
            for(MatOfPoint points : contours){
                Rect r = Imgproc.boundingRect(points);

                Point center = new Point(r.x + r.width / 2, r.y + r.height / 2);
                int squareWidth = r.width > r.height ? r.width + 5: r.height + 5;
                Rect sub = new Rect((int)(center.x) - squareWidth / 2, (int)center.y - squareWidth / 2, squareWidth, squareWidth);
                if(sub.x < 0 || sub.y < 0 || sub.x + sub.width > mRgba.cols() || sub.height + sub.y > mRgba.rows() || sub.width < 0 || sub.height < 0)
                    continue;
                Log.d("RECT: ", Integer.toString(r.width) + " , " + Integer.toString(r.height));
                Mat submat = mRgba.submat(sub);
                Imgproc.resize(submat, submat, DrawView.inputSize);
                Core.rotate(submat, submat, Core.ROTATE_90_CLOCKWISE);
//                Bitmap bitmap = Bitmap.createBitmap((int)DrawView.inputSize.width, (int)DrawView.inputSize.height, Bitmap.Config.ARGB_8888);;
//                Utils.matToBitmap(submat, bitmap);
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
                Imgproc.rectangle(mRgba, new Point(r.x, r.y), new Point(r.x + r.width, r.y + r.height), new Scalar(255, 0, 0, 255), 3);
//                Log.d("LETTER: ", DrawView.mapOutputToChar(maxIndex));
                Imgproc.putText(mRgba, DrawView.mapOutputToChar(maxIndex), new Point(20, 300), Core.FONT_HERSHEY_PLAIN, 15.0, new Scalar(200, 200, 255));
            }
        }

        // Rotate mRgba 90 degrees
        //Core.transpose(mRgba, mRgbaT);
        //Imgproc.resize(mRgbaT, mRgbaF, mRgbaF.size(), 0,0, 0);
        //Core.flip(mRgbaF, mRgba, 1 );
//        Point center = new Point(1920 / 2.0, 1080 / 2.0);//new Point(mRgba.width() / 2.0, mRgba.height() / 2.0);
//        float size = 0.1f;
//        Imgproc.rectangle(mRgba, new Point(center.x * (1 - size), center.y * (1 - size)), new Point(center.x * (1 + size), center.x * (1 + size)), new Scalar(255, 0, 255), 3);
        return mRgba; // This function must return
    }
}