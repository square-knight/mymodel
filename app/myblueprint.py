#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@User    : frank 
@Time    : 2019/7/20 10:16
@File    : myblueprint.py
@Email  : frank.chang@xinyongfei.cn
"""
# 定义一个蓝图
from flask import Blueprint

web: Blueprint = Blueprint('web', __name__)

if __name__ == '__main__':
    pass
