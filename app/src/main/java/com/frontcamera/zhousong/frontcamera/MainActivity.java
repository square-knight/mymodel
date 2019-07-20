package com.frontcamera.zhousong.frontcamera;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.os.Message;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.KeyEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.Switch;
import android.widget.TextView;
import android.os.Handler;
import android.widget.Toast;

import com.frontcamera.zhousong.constant.Constant;

import java.lang.ref.WeakReference;

public class MainActivity extends Activity {
    Context context = MainActivity.this;
    SurfaceView surfaceView;
    TextView numberTextView;
    EditText uriEditText;
    EditText collectEditText;
    Switch collectSwitch;
    Button trainButton;
    CameraSurfaceHolder mCameraSurfaceHolder = new CameraSurfaceHolder();
    private UIHandler uiHandler;
    private long firstTime;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        uiHandler=new UIHandler(this);
        initView();
        addListener();
    }

    public void initView()
    {
        setContentView(R.layout.activity_main);
        surfaceView = (SurfaceView) findViewById(R.id.surfaceView1);
        View photoFrameLayout = findViewById(R.id.photoFrameLayout);
        resizeView(photoFrameLayout);
        mCameraSurfaceHolder.setCameraSurfaceHolder(context,surfaceView);
        numberTextView = (TextView) findViewById(R.id.numberTextView);
        uriEditText = (EditText) findViewById(R.id.uriEditText);
        uriEditText.setText(Constant.URL_UPLOAD);
        collectEditText = (EditText) findViewById(R.id.collectEditText);
        collectSwitch = (Switch) findViewById(R.id.collectSwitch);
        trainButton = (Button) findViewById(R.id.trainButton);
    }
    private void addListener(){
        uriEditText.setOnFocusChangeListener(new View.OnFocusChangeListener(){
            @Override
            public void onFocusChange(View v, boolean hasFocus) {
                if(!hasFocus){
                    Constant.url = uriEditText.getText().toString();
                }
            }
        });
        collectSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if(buttonView.isPressed()){
                    // 每次 setChecked 时会触发onCheckedChanged 监听回调，而有时我们在设置setChecked后不想去自动触发 onCheckedChanged 里的具体操作, 即想屏蔽掉onCheckedChanged;加上此判断
                    showToast((isChecked ? "开" : "关") + "图片组:" + collectEditText.getText());
                }
            }
        });
        trainButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                showToast("按钮文字为:" + trainButton.getText());
            }
        });


    }
    private void resizeView(View view){
        WindowManager wm = (WindowManager) this.getSystemService(Context.WINDOW_SERVICE);
        DisplayMetrics dm = new DisplayMetrics();
        wm.getDefaultDisplay().getMetrics(dm);
        int width = dm.widthPixels;         // 屏幕宽度（像素）
//        int height = dm.heightPixels;       // 屏幕高度（像素）
//        float density = dm.density;         // 屏幕密度（0.75 / 1.0 / 1.5）
//        int densityDpi = dm.densityDpi;     // 屏幕密度dpi（120 / 160 / 240）
        // 屏幕宽度算法:屏幕宽度（像素）/屏幕密度
//        int screenWidth = (int) (width / density);  // 屏幕宽度(dp)
//        int screenHeight = (int) (height / density);// 屏幕高度(dp)
        ViewGroup.LayoutParams linearParams = view.getLayoutParams();
        linearParams.height = width - 20;
        view.setLayoutParams(linearParams);
    }
    public UIHandler getUiHandler(){
        return uiHandler;
    }
    public static class UIHandler extends Handler {
        WeakReference<MainActivity> activity;
        public UIHandler(MainActivity activity){
            this.activity=new WeakReference<MainActivity>(activity);
        }
        @Override
        public void handleMessage(Message msg) {
            super.handleMessage(msg);
            if(null==this.activity){
                return;
            }
            this.activity.get().numberTextView.setText((String)msg.obj);


        }
    }

    @Override
    public boolean onKeyDown(int keyCode, KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_BACK && event.getAction() == KeyEvent.ACTION_DOWN) {
            long secondTime = System.currentTimeMillis();
            if (secondTime - firstTime > 2000) {
                showToast("再戳一下退出哦");
                firstTime = secondTime;
                return true;
            } else{
                finish();
            }
        }
        return super.onKeyDown(keyCode, event);
    }
    private void showToast(String text){
        Toast toast=Toast.makeText(getApplicationContext(), text, Toast.LENGTH_SHORT);
        toast.setText(text);
        toast.show();
    }
}
