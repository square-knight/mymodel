#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@User    : frank 
@Time    : 2019/7/19 09:33
@File    : log.py
@Email  : frank.chang@xinyongfei.cn
"""
import os
import logging.config

LOG_BACKUP_COUNT = 50
LOG_ROTATING_FILE_MODE = 'time'
LOG_DIR = '../logs/'

logger = logging.getLogger('log')

default_logging_config = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'console': {
            'format': '[%(asctime)s][%(levelname)s/%(process)s] %(name)s '
                      '%(filename)s:%(funcName)s:%(lineno)d | %(message)s',
        },
    },

    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'console',
        },
    },

    'loggers': {
        '': {
            'level': 'ERROR',
            'handlers': ['console', ],
            'propagate': False
        },
    },

    'root': {
        'level': 'INFO',
        'handlers': ['console', 'info_file_handler', 'error_file_handler']
    }
}

time_info_file_handler = {
    'class': 'logging.handlers.TimedRotatingFileHandler',
    'level': 'INFO',
    'formatter': 'console',
    'when': 'D',
    'encoding': 'utf8',
    'backupCount': LOG_BACKUP_COUNT,
}

time_error_file_handler = {
    'class': 'logging.handlers.TimedRotatingFileHandler',
    'level': 'ERROR',
    'formatter': 'console',
    'when': 'D',
    'encoding': 'utf8',
    'backupCount': LOG_BACKUP_COUNT,
}

size_info_file_handler = {
    'class': 'logging.handlers.RotatingFileHandler',
    'level': 'INFO',
    'formatter': 'console',
    # 当达到 250MB 时分割日志
    'maxBytes': 262144000,
    # 最多保留 100 份文件
    'backupCount': LOG_BACKUP_COUNT,
    'encoding': 'utf8'
}

size_error_file_handler = {
    'class': 'logging.handlers.RotatingFileHandler',
    'level': 'ERROR',
    'formatter': 'console',
    'maxBytes': 262144000,
    'backupCount': LOG_BACKUP_COUNT,
    'encoding': 'utf8'
}


def configure_logging(log_path=LOG_DIR):
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    info_file_name = os.path.join(log_path, 'info.log')
    error_file_name = os.path.join(log_path, 'error.log')

    if LOG_ROTATING_FILE_MODE == 'size':
        info_file_handler = size_info_file_handler
        error_file_handler = size_error_file_handler
    else:
        info_file_handler = time_info_file_handler
        error_file_handler = time_error_file_handler

    info_file_handler.update(filename=info_file_name)
    error_file_handler.update(filename=error_file_name)

    default_logging_config['handlers']['info_file_handler'] = info_file_handler
    default_logging_config['handlers']['error_file_handler'] = error_file_handler
    logging.config.dictConfig(default_logging_config)


if __name__ == '__main__':
    configure_logging()
    logger = logging.getLogger('logger')
    logger.info('test logger')
