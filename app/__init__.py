#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@User    : frank 
@Time    : 2019/7/18 20:10
@File    : __init__.py
@Email  : frank.chang@xinyongfei.cn

references:
    http://docs.jinkan.org/docs/flask/patterns/fileuploads.html
"""
import logging
from flask import Flask

from app.main import readImageFromDisk
from util.log import configure_logging

configure_logging("../logs/")

logger = logging.getLogger(__name__)


def register_blueprint(app):
    from app.web.upload_service import web
    app.register_blueprint(web)
    pass


def create_app():
    app = Flask(__name__)

    # 加载app配置文件
    app.config.from_object('config.DB')

    # 注册蓝图
    register_blueprint(app)

    return app
