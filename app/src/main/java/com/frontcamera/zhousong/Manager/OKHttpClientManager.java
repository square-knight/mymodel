package com.frontcamera.zhousong.Manager;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSession;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;

import okhttp3.FormBody;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class OKHttpClientManager {

    private String reqUrl;
    private int connectTimeout;
    private int readTimeOut;

    public static MediaType MEDIATYPE_X_WWW_FROM_URL_ENCORED = MediaType.parse("application/x-www-form-urlencoded");
    public static MediaType MEDIATYPE_JSON = MediaType.parse("application/json");

    private static Map<String, OKHttpClientManager> okHttpClientManages = new HashMap<String, OKHttpClientManager>();
    //    private static OKHttpClientManage okHttpClientManage;
    private OkHttpClient client;

    private OKHttpClientManager(String reqUrl, int connectTimeout, int readTimeOut) {
        super();
        this.reqUrl = reqUrl;
        this.connectTimeout = connectTimeout;
        this.readTimeOut = readTimeOut;
        initClient();
    }
    private OKHttpClientManager(String reqUrl, int connectTimeout, int readTimeOut, boolean unsafe) {
        super();
        this.reqUrl = reqUrl;
        this.connectTimeout = connectTimeout;
        this.readTimeOut = readTimeOut;
        if(unsafe){
            initUnsafeClient();
        }else {
            initClient();
        }

    }
    public static OKHttpClientManager getOkHttpClientManage(String reqUrl) {
        return getOkHttpClientManage(reqUrl,10000,5000);
    }
    public static OKHttpClientManager getOkHttpClientManage(String reqUrl, int connectTimeout, int readTimeOut) {
        OKHttpClientManager okHttpClientManage = okHttpClientManages.get(reqUrl);
        if (okHttpClientManage == null) {
            okHttpClientManage = new OKHttpClientManager(reqUrl, connectTimeout, readTimeOut);
            okHttpClientManages.put(reqUrl, okHttpClientManage);
        }
        return okHttpClientManage;
    }
    public static OKHttpClientManager getOkHttpClientManageUnsafe(String reqUrl, int connectTimeout, int readTimeOut) {
        OKHttpClientManager okHttpClientManage = okHttpClientManages.get(reqUrl);
        if (okHttpClientManage == null) {
            okHttpClientManage = new OKHttpClientManager(reqUrl, connectTimeout, readTimeOut,true);
            okHttpClientManages.put(reqUrl, okHttpClientManage);
        }
        return okHttpClientManage;
    }
    private void initClient() {
        if (client == null) {
            client = new OkHttpClient().newBuilder()
                    .connectTimeout(connectTimeout, TimeUnit.MILLISECONDS)
                    .readTimeout(readTimeOut, TimeUnit.MILLISECONDS)
                    .build();
        }
    }
    private void initUnsafeClient() {
        if (client == null) {
            client = getUnsafeOkHttpClient();
        }
    }
    public String post(String params, MediaType mediaType) throws Exception {
        try {
//            logger.info("reqUrl=" + reqUrl + " ,requestBody=" + params);
            RequestBody body = RequestBody.create(mediaType, params);
            Request request = new Request.Builder()
                    .url(reqUrl)
                    .post(body)
                    .addHeader("cache-control", "no-cache")
                    .build();
            Response response = client.newCall(request).execute();
//            logger.info("response=" + response.toString());
            if (!response.isSuccessful()) {
                String excepMsg = response.code() + ":" + response.message();
                response.body().close();
                throw new RuntimeException(excepMsg);
            }
            return response.body().string();
        } catch (Exception e) {
//            logger.error("", e);
            throw e;
        }
    }
    public String post(byte[] params) throws Exception {
        try {
//            logger.info("reqUrl=" + reqUrl + " ,requestBody=" + params);
            RequestBody body = RequestBody.create(MediaType.parse("application/octet.stream"), params);
            Request request = new Request.Builder()
                    .url(reqUrl)
                    .post(body)
                    .addHeader("cache-control", "no-cache")
                    .build();
            Response response = client.newCall(request).execute();
//            logger.info("response=" + response.toString());
            if (!response.isSuccessful()) {
                String excepMsg = response.code() + ":" + response.message();
                response.body().close();
                throw new RuntimeException(excepMsg);
            }
            return response.body().string();
        } catch (Exception e) {
//            logger.error("", e);
            throw e;
        }
    }
    public String postFrom(Map<String,String> params) throws Exception {
        try {
//            logger.info("reqUrl=" + reqUrl + " ,requestBody=" + params);
            FormBody.Builder builder = new FormBody.Builder();
            for (Map.Entry<String, String> entry: params.entrySet()){
                builder.add(entry.getKey(),entry.getValue());
            }
            FormBody formBody = builder.build();
            Request request = new Request.Builder()
                    .url(reqUrl)
                    .post(formBody)
                    .addHeader("cache-control", "no-cache")
                    .build();

            Response response = client.newCall(request).execute();
//            logger.info("response=" + response.toString());
            if (!response.isSuccessful()) {
                String excepMsg = response.code() + ":" + response.message();
                response.body().close();
                throw new RuntimeException(excepMsg);
            }
            return response.body().string();
        } catch (Exception e) {
//            logger.error("", e);
            throw e;
        }
    }
    public String put(String params, MediaType mediaType) throws Exception {
        try {
//            logger.info("reqUrl=" + reqUrl + " ,requestBody=" + params);
            RequestBody body = RequestBody.create(mediaType, params);
            Request.Builder builder = new Request.Builder()
                    .url(reqUrl)
                    .put(body)
                    .addHeader("cache-control", "no-cache");
            Request request = builder.build();
            Response response = client.newCall(request).execute();
//            logger.info("response=" + response.toString());
            if (!response.isSuccessful()) {
                String excepMsg = response.code() + ":" + response.message();
                response.body().close();
                throw new RuntimeException(excepMsg);
            }
            return response.body().string();
        } catch (Exception e) {
//            logger.error("", e);
            throw e;
        }
    }
    public String put(String params, MediaType mediaType,Map<String,String> headers) throws Exception {
        try {
//            logger.info("reqUrl=" + reqUrl + " ,requestBody=" + params);
            RequestBody body = RequestBody.create(mediaType, params);
            Request.Builder builder = new Request.Builder()
                    .url(reqUrl)
                    .put(body)
                    .addHeader("cache-control", "no-cache");
            for (Map.Entry<String,String> entry:
            headers.entrySet()) {
                builder.addHeader(entry.getKey(),entry.getValue());
            }
//            headers.forEach((key, value) ->{
//                builder.addHeader(key,value);
//            });
            Request request = builder.build();
            Response response = client.newCall(request).execute();
//            logger.info("response=" + response.toString());
            if (!response.isSuccessful()) {
                String excepMsg = response.code() + ":" + response.message();
                response.body().close();
                throw new RuntimeException(excepMsg);
            }
            return response.body().string();
        } catch (Exception e) {
//            logger.error("", e);
            throw e;
        }
    }
    public String post(String params, MediaType mediaType,Map<String,String> headers) throws Exception {
        try {
//            logger.info("reqUrl=" + reqUrl + " ,requestBody=" + params);
            RequestBody body = RequestBody.create(mediaType, params);
            Request.Builder builder = new Request.Builder()
                    .url(reqUrl)
                    .post(body)
                    .addHeader("cache-control", "no-cache");
            for (Map.Entry<String,String> entry:
                    headers.entrySet()) {
                builder.addHeader(entry.getKey(),entry.getValue());
            }
//            headers.forEach((key, value) ->{
//                builder.addHeader(key,value);
//            });
            Request request = builder.build();
            Response response = client.newCall(request).execute();
//            logger.info("response=" + response.toString());
            if (!response.isSuccessful()) {
                String excepMsg = response.code() + ":" + response.message();
                response.body().close();
                throw new RuntimeException(excepMsg);
            }
            return response.body().string();
        } catch (Exception e) {
//            logger.error("", e);
            throw e;
        }
    }

    public String delete(String params, MediaType mediaType) throws Exception {
        try {
//            logger.info("reqUrl=" + reqUrl + " ,requestBody=" + params);
            RequestBody body = RequestBody.create(mediaType, params);
            Request request = new Request.Builder()
                    .url(reqUrl)
                    .delete(body)
                    .addHeader("cache-control", "no-cache")
                    .build();
            Response response = client.newCall(request).execute();
//            logger.info("response=" + response.toString());
            if (!response.isSuccessful()) {
                String excepMsg = response.code() + ":" + response.message();
                response.body().close();
                throw new RuntimeException(excepMsg);
            }
            return response.body().string();
        } catch (Exception e) {
//            logger.error("", e);
            throw e;
        }
    }

    public String get(String params) throws Exception {
        try {
//            logger.info("requestBody=" + params);
            Request request = new Request.Builder()
                    .url(reqUrl)
                    .get()
                    .addHeader("cache-control", "no-cache")
                    .build();
            Response response = client.newCall(request).execute();
//            logger.info("responseBody=" + response.toString() + ", body=" + response.body());
            if (!response.isSuccessful()) {
                String excepMsg = response.code() + ":" + response.message();
                response.body().close();
                throw new RuntimeException(excepMsg);
            }
            return response.body().string();
        } catch (Exception e) {
//            logger.error("请求异常！reqUrl=" + reqUrl, e);
            throw e;
        }
    }

    public String get(String params, Map<String,String> headers) throws Exception {
        try {
//            logger.info("requestBody=" + params);
            Request.Builder builder = new Request.Builder()
                    .url(reqUrl)
                    .get()
                    .addHeader("cache-control", "no-cache");
            for (Map.Entry<String,String> entry:
                    headers.entrySet()) {
                builder.addHeader(entry.getKey(),entry.getValue());
            }
//            headers.forEach((key, value) ->{
//                builder.addHeader(key,value);
//            });
            Request request = builder.build();
            Response response = client.newCall(request).execute();
//            logger.info("responseBody=" + response.toString() + ", body=" + response.body());
            if (!response.isSuccessful()) {
                String excepMsg = response.code() + ":" + response.message();
                response.body().close();
                throw new RuntimeException(excepMsg);
            }
            return response.body().string();
        } catch (Exception e) {
//            logger.error("请求异常！reqUrl=" + reqUrl, e);
            throw e;
        }
    }

    private OkHttpClient getUnsafeOkHttpClient() {
        try {
            // Create a trust manager that does not validate certificate chains
            final TrustManager[] trustAllCerts = new TrustManager[] {
                    new X509TrustManager() {
                        @Override
                        public void checkClientTrusted(java.security.cert.X509Certificate[] chain, String authType) {
                        }

                        @Override
                        public void checkServerTrusted(java.security.cert.X509Certificate[] chain, String authType) {
                        }

                        @Override
                        public java.security.cert.X509Certificate[] getAcceptedIssuers() {
                            return new java.security.cert.X509Certificate[]{};
                        }
                    }
            };

            // Install the all-trusting trust manager
            final SSLContext sslContext = SSLContext.getInstance("SSL");
            sslContext.init(null, trustAllCerts, new java.security.SecureRandom());
            // Create an ssl socket factory with our all-trusting manager
            final SSLSocketFactory sslSocketFactory = sslContext.getSocketFactory();

            OkHttpClient.Builder builder = new OkHttpClient.Builder();
            builder.sslSocketFactory(sslSocketFactory);
            builder.hostnameVerifier(new HostnameVerifier() {
                @Override
                public boolean verify(String hostname, SSLSession session) {
                    return true;
                }
            });

            OkHttpClient okHttpClient = builder.build();
            return okHttpClient;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}

