# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     __init__.py
   Description :
   Author :       chenhao
   date：          2020/7/29
-------------------------------------------------
   Change Activity:
                   2020/7/29:
-------------------------------------------------
"""
import logging
import pydantic
import warnings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s] [%(levelname)s]: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')

pydantic.BaseConfig.arbitrary_types_allowed = True

warnings.filterwarnings('ignore')
