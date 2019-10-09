# 手指个数识别-server

## 如何将Server跑起来？
- 本地启动redis
- 把/config/APP.py的project_path修改为本地项目所在目录
- 启动start_model.py
- Android机访问/download/ohmyfinger.apk接口下载安装Android客户端

## client
[Android app](https://github.com/square-knight/OhMyFinger)
安装Android app授予网络和摄像机权限
app有图像收集和预测两种功能

## 数据集
[手指图片](https://github.com/square-knight/finger_train_set)
