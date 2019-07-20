package com.frontcamera.zhousong.frontcamera;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Message;
import android.util.Log;

import com.frontcamera.zhousong.Manager.OKHttpClientManager;
import com.frontcamera.zhousong.constant.Constant;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Created by zhousong on 2016/9/28.
 * 单独的任务类。继承AsyncTask，来处理从相机实时获取的耗时操作
 */
public class FaceTask extends AsyncTask{
    private byte[] mData;
    Camera mCamera;
    MainActivity mainActivity;
    private static final String TAG = "CameraTag";
    //构造函数
    FaceTask(byte[] data , Camera camera)
    {
        this.mData = data;
        this.mCamera = camera;

    }
    @Override
    protected Object doInBackground(Object[] params) {
        Camera.Parameters parameters = mCamera.getParameters();
        int imageFormat = parameters.getPreviewFormat();
        int w = parameters.getPreviewSize().width;
        int h = parameters.getPreviewSize().height;
        String text = "";
        Message message =Message.obtain();
        Rect rect = new Rect(0, 0, w, h);
        YuvImage yuvImg = new YuvImage(mData, imageFormat, w, h, null);
        try {
            ByteArrayOutputStream outputstream = new ByteArrayOutputStream();
            yuvImg.compressToJpeg(rect, 100, outputstream);
            Bitmap rawbitmap = BitmapFactory.decodeByteArray(outputstream.toByteArray(), 0, outputstream.size());
            Bitmap cuttedBitmap = Bitmap.createBitmap(rawbitmap, 0, 0, h, h);
            Bitmap scaledBitmap = Bitmap.createScaledBitmap(cuttedBitmap, 64, 64, true);
            ByteArrayOutputStream jpgOutputStream = new ByteArrayOutputStream();
            scaledBitmap.compress(Bitmap.CompressFormat.JPEG,100,jpgOutputStream);

            byte[] bytes = jpgOutputStream.toByteArray();
            Log.i(TAG, "onPreviewFrame: rawbitmap:" + rawbitmap.toString());
            OKHttpClientManager okHttpClientManage = OKHttpClientManager.getOkHttpClientManage(Constant.url);
            String post = okHttpClientManage.post(bytes);
            text = post;
//            okHttpClientManage.post()
            //若要存储可以用下列代码，格式为jpg
            /* BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(Environment.getExternalStorageDirectory().getPath()+"/fp.jpg"));
            img.compressToJpeg(rect, 100, bos);
            bos.flush();
            bos.close();
            mCamera.startPreview();
            */
        }
        catch (Exception e)
        {
            text = "获取相机实时数据失败" + e.getLocalizedMessage();
            Log.e(TAG, "onPreviewFrame: 获取相机实时数据失败" + e.getLocalizedMessage());
        }
        message.what = 10000;
        message.obj = text;
        mainActivity.getUiHandler().sendMessage(message);
        return null;
    }

//    public static void main(String[] args) throws Exception {
//
//        OKHttpClientManager okHttpClientManage = OKHttpClientManager.getOkHttpClientManage(Constant.URL_UPLOAD);
//        okHttpClientManage.post(getBytes("/Users/doom/Documents/cat1.jpeg"));
//    }
//    /**
//     * 将文件转为byte[]
//     * @param filePath 文件路径
//     * @return
//     */
//    public static byte[] getBytes(String filePath){
//        File file = new File(filePath);
//        ByteArrayOutputStream out = null;
//        try {
//            FileInputStream in = new FileInputStream(file);
//            out = new ByteArrayOutputStream();
//            byte[] b = new byte[1024];
//            int i = 0;
//            while ((i = in.read(b)) != -1) {
//
//                out.write(b, 0, b.length);
//            }
//            out.close();
//            in.close();
//        } catch (FileNotFoundException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
//        } catch (IOException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
//        }
//        byte[] s = out.toByteArray();
//        return s;
//
//    }
}
