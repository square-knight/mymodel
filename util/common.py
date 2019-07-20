#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@User    : frank 
@Time    : 2019/7/20 10:10
@File    : common.py
@Email  : frank.chang@xinyongfei.cn
"""

import time


def now_to_timestamp(digits=10):
    """
    生成 时间戳
    :param digits:
    :return: int
    """
    time_stamp = time.time()
    digits = 10 ** (digits -10)
    time_stamp = int(round(time_stamp*digits))
    return time_stamp



if __name__ == '__main__':
    t  =  now_to_timestamp()
    print(t)
    print(type(t))

    pass
    
