#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@User    : frank 
@Time    : 2019/7/18 20:10
@File    : __init__.py
@Email  : frank.chang@xinyongfei.cn
"""
from flask import Flask
from flask import request

from app.main import readImageFromDisk
from app.main import predict
import numpy as np

app = Flask(__name__)


def run():
    app.run(port=1880)


@app.route('/')
@app.route('/index')
def index():
    return "This is Index !"


@app.route('/predict', methods=["POST", "GET"])
def predict_model():
    request_json = request.json




    # 1 download picture

    # 2  readImageFromDisk  读文件

    image = readImageFromDisk(path="./images/")

    # predict
    img = np.reshape(image, (1, 64, 64, 3))


    y_predict = predict(img)

    print(image.shape)
    print("y_predict:\n", y_predict)


if __name__ == '__main__':
    run()
    pass
