#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@User    : frank 
@Time    : 2019/7/18 20:10
@File    : start_model.py
@Email  : frank.chang@xinyongfei.cn
"""
from app import create_app

app = create_app()


@app.route('/')
@app.route('/index')
def index():
    return "This is Index !"


def run():
    # app.run(host='192.168.10.57',port=1880)
    app.run(host='0.0.0.0', port=9003)


if __name__ == '__main__':
    run()
    pass
