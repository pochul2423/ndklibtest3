package com.example.ndklibtest3;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.content.DialogInterface;
import android.os.Bundle;
import android.annotation.TargetApi;
import android.content.pm.PackageManager;
import android.os.Build;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.Collections;
import java.util.List;

import static android.Manifest.permission.CAMERA;


//https://webnautes.tistory.com/923

class Paramclass{
    int x;
    int y;
    int radius;
}

public class MainActivity extends AppCompatActivity
        implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "opencv";
    private Mat matInput;
    private Mat matDebug;
    private float result=0;
    private CameraBridgeViewBase mOpenCvCameraView;
    int x=0,y=0,radius=0;

    //public native void FindCircle(long matAddrInput, int x, int y, int radius);
    public native float FindCircle(long matAddrInput, Paramclass param);
    public native float AnalogGageIndicatorVal(long matAddrInput, int x, int y, int radius, float resizeRatio, long matDebug);


    private static int cnt=0;

    static {
        System.loadLibrary("opencv_java4");
        System.loadLibrary("native-lib");

    }



    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        Button button1 = (Button) findViewById(R.id.button4) ;
        button1.setOnClickListener(new Button.OnClickListener() {
            @Override
            public void onClick(View view) {
            if(radius>0)
            {
                Paramclass info = new Paramclass();
                info.x = 0;
                info.y=0;
                info.radius=0;

                float resizedRatio = FindCircle(matInput.getNativeObjAddr(), info);
                result = AnalogGageIndicatorVal(matInput.getNativeObjAddr(), x,y,radius, resizedRatio, matDebug.getNativeObjAddr());
                Toast.makeText(MainActivity.this,String.valueOf(result),Toast.LENGTH_SHORT).show();
            }else
            {
                Toast.makeText(MainActivity.this,"카메라에 게이지를 정확히 위치해주세요.",Toast.LENGTH_SHORT).show();
            }



                // TODO : click event
            }
        });

        mOpenCvCameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(0); // front-camera(1),  back-camera(0)
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
            Log.d(TAG, "onResume :: Internal OpenCV library not found.");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "onResum :: OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    public void onDestroy() {
        super.onDestroy();

        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        matInput = inputFrame.rgba();

        if ( matDebug == null )
            matDebug = new Mat(matInput.rows(), matInput.cols(), matInput.type());

        Paramclass info = new Paramclass();
        info.x = 0;
        info.y=0;
        info.radius=0;


        //x=y=radius=100;


        float resizedRatio = FindCircle(matInput.getNativeObjAddr(), info);

        x = info.x;
        y = info.y;
        radius = info.radius;


        //Imgproc.putText(matInput, Integer.toString(x), new Point(100,300), 1, 5, new Scalar(255,0,0), 5);
        //Imgproc.putText(matInput, Integer.toString(y), new Point(100,360), 1, 5, new Scalar(255,0,0), 5);
        /*Imgproc.putText(matInput, Float.toString(radius), new Point(100,420), 1, 5, new Scalar(255,0,0), 5);
        Imgproc.putText(matInput, Float.toString(resizedRatio), new Point(100,480), 1, 5, new Scalar(255,0,0), 5);
        Imgproc.putText(matInput, Integer.toString(matInput.cols()), new Point(100,540), 1, 5, new Scalar(255,0,0), 5);
        Imgproc.putText(matInput, Integer.toString(matInput.rows()), new Point(100,600), 1, 5, new Scalar(255,0,0), 5);
*/

        if(radius>0) {
            result = AnalogGageIndicatorVal(matInput.getNativeObjAddr(), x,y,radius, resizedRatio, matDebug.getNativeObjAddr());
        }

        x /= resizedRatio;
        y /= resizedRatio;
        radius /= resizedRatio;

        Imgproc.circle(matInput, new Point(x,y), radius, new Scalar(255,0,0), 3);

        Imgproc.putText(matInput, Float.toString(result), new Point(100,200), 1, 5, new Scalar(255,0,0), 5);


        cnt++;

        return matInput;

//        return matDebug;

    }


    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(mOpenCvCameraView);
    }


    //여기서부턴 퍼미션 관련 메소드
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 200;


    protected void onCameraPermissionGranted() {
        List<? extends CameraBridgeViewBase> cameraViews = getCameraViewList();
        if (cameraViews == null) {
            return;
        }
        for (CameraBridgeViewBase cameraBridgeViewBase: cameraViews) {
            if (cameraBridgeViewBase != null) {
                cameraBridgeViewBase.setCameraPermissionGranted();
            }
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        boolean havePermission = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
                havePermission = false;
            }
        }
        if (havePermission) {
            onCameraPermissionGranted();
        }
    }

    @Override
    @TargetApi(Build.VERSION_CODES.M)
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST_CODE && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            onCameraPermissionGranted();
        }else{
            showDialogForPermission("앱을 실행하려면 퍼미션을 허가하셔야합니다.");
        }
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }


    @TargetApi(Build.VERSION_CODES.M)
    private void showDialogForPermission(String msg) {

        AlertDialog.Builder builder = new AlertDialog.Builder( MainActivity.this);
        builder.setTitle("알림");
        builder.setMessage(msg);
        builder.setCancelable(false);
        builder.setPositiveButton("예", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface dialog, int id){
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("아니오", new DialogInterface.OnClickListener() {
            public void onClick(DialogInterface arg0, int arg1) {
                finish();
            }
        });
        builder.create().show();
    }
}