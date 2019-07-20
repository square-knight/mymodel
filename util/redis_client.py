#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@User    : frank 
@Time    : 2019/7/20 09:48
@File    : redis_client.py
@Email  : frank.chang@xinyongfei.cn
"""
import redis

from config.DB import REDIS_CONFIG


class RedisClient:
    """
    单例模式
    """
    _instance = None

    @classmethod
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            pool = redis.ConnectionPool(**REDIS_CONFIG, db=0)
            cls._instance = redis.StrictRedis(connection_pool=pool)

        return cls._instance


redis_client = RedisClient()


class IdGenerator:

    def __init__(self, key, client):
        self.key = key

        # redis 客户端.
        self.client = client

    def init(self, n):
        self.client.set(self.key, n)

    def gen_id(self):
        """
        返回一个 id
        :return:  int
        """
        new_id = self.client.incr(self.key)
        return new_id


generator = IdGenerator('mymodel:images:id', redis_client)
generator.init(1000)

if __name__ == '__main__1111':
    generator = IdGenerator('test', redis_client)

    generator.init(100)

    print(generator.gen_id())
    print(generator.gen_id())
    print(generator.gen_id())
    print(generator.gen_id())
    pass
