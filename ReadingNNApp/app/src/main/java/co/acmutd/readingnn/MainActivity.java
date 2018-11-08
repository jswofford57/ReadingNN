package co.acmutd.readingnn;

import android.content.Intent;
import android.support.annotation.Nullable;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity {
    private static String TAG = "ReadingNN";
    private Button clearButton, analyzeButton;
    private DrawView drawView;
    private TextView result;
    private RelativeLayout parent;
    private FloatingActionButton cameraButton;

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
                        }
                    });
                    cameraButton.setOnClickListener(new Button.OnClickListener(){
                        @Override
                        public void onClick(View view){
                            Intent myIntent = new Intent(MainActivity.this, CameraActivity.class);
                            MainActivity.this.startActivity(myIntent);
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
        drawView = new DrawView(this);
        parent.addView(drawView);
        cameraButton = findViewById(R.id.cameraButton);
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