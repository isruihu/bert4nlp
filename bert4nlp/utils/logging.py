"""
Imported from https://github.com/grayhong/bias-contrastive-learning/blob/master/debias/utils/logging.py
"""

import logging
from logging.config import dictConfig
import time

# [%(asctime)s] [%(levelname)s]

def set_logging(logger_name, level, work_dir):
    LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": f"%(message)s"
            },
        },
        "handlers": {
            "console": {
                "level": f"{level}",
                "class": "logging.StreamHandler",
                'formatter': 'simple',
            },
            'file': {
                'level': f"{level}",
                'formatter': 'simple',
                'class': 'logging.FileHandler',
                'filename': f'{work_dir if work_dir is not None else "."}/train.log',
                'mode': 'a',
            },
        },
        "loggers": {
            "": {
                "level": f"{level}",
                "handlers": ["console", "file"] if work_dir is not None else ["console"],
            },
        },
    }
    dictConfig(LOGGING)

    date = time.strftime('%Y/%m/%d %H:%M')
    logging.info(f"Log level set to: {level} at {date}")