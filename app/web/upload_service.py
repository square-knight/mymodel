#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@User    : frank 
@Time    : 2019/7/20 10:18
@File    : upload_service.py
@Email  : frank.chang@xinyongfei.cn
"""
import logging
import os
from io import BytesIO
from flask import request

import numpy as np
from PIL import Image

from app.main import predict
from app.main import readImageFromDisk

from config.APP import images_path, images_path_train
from util.filename import gen_image_filename

from app.myblueprint import web

logger = logging.getLogger(__name__)


@web.route('/upload1111_test', methods=['GET', 'POST'])
def upload_file():
    logger.info(f"request: {request}")

    file = request.get_data()
    if not file:
        return "no file error !"
    image = Image.open(BytesIO(file))
    # print(f"image:{image}")
    # logger.info(f"content_type: {request.content_type}")

    filename = gen_image_filename()

    absolute_path = os.path.join(images_path, filename)
    image.save(absolute_path)

    message = f"图片已经保存. path: {absolute_path},filename:{filename}"
    logger.info(message)
    return message


@web.route('/predict', methods=["POST", "GET"])
def predict_model():
    logger.info(f"request: {request}")

    logger.info(request.headers)
    file = request.get_data()
    if not file:
        return "no file error !"
    image = Image.open(BytesIO(file))
    # print(f"image:{image}")
    # logger.info(f"content_type: {request.content_type}")

    filename = gen_image_filename()

    absolute_path = os.path.join(images_path, filename)
    image.save(absolute_path)

    message = f"图片已经保存. path: {absolute_path},filename:{filename}"
    logger.info(message)

    # 1 download picture

    # 2  readImageFromDisk  读文件

    image = readImageFromDisk(path=absolute_path)

    # predict
    img = np.reshape(image, (1, 64, 64, 3))

    y_predict = predict(img)

    # print(image.shape)
    logger.info(f"y_predict: {y_predict}")
    return str(y_predict)


@web.route('/collect', methods=["POST", "GET"])
def collect_img():
    y = request.headers['y']
    yi = int(y)
    if yi < 0 or yi > 5:
        return "not 0-5 error!"
    file = request.get_data()
    if not file:
        return "no file error !"
    image = Image.open(BytesIO(file))

    filename = gen_image_filename()
    filename = y + "_" + filename
    absolute_path = os.path.join(images_path_train, filename)
    image.save(absolute_path)

    message = f"图片已经保存. path: {absolute_path},filename:{filename}"
    logger.info(message)

    return "ok"


@web.route('/train', methods=["POST", "GET"])
def train():

    return "ok"


@web.route('/upload11111', methods=['GET', 'POST'])
def upload_file111():
    if request.method == 'POST':
        file = request.files['file']

        # if file and allowed_file(file.filename):
        #
        #     filename = secure_filename(file.filename)
        #
        #     logger.info(f"filename:{filename}")
        #
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #     return 'upload file sucesss.'
        # else:
        #     logger.error('error filename')
        #     return 'invalid  filename !'

    return "hello world"


if __name__ == '__main__':
    pass
