#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@User    : frank 
@Time    : 2019/7/20 10:58
@File    : filename.py
@Email  : frank.chang@xinyongfei.cn
"""
from util.common import now_to_timestamp
from util.redis_client import generator


def gen_image_filename():
    """
    返回 文件的一个名称
    return  timestape + 自增id  + .png

    :return:  str 15635911121117.png

    """
    timestamp = str(now_to_timestamp())
    new_id = str(generator.gen_id())
    filename = timestamp + new_id + '.jpg'
    return filename


if __name__ == '__main__':
    pass
