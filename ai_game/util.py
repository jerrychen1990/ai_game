# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     util
   Description :
   Author :       chenhao
   date：          2020/8/5
-------------------------------------------------
   Change Activity:
                   2020/8/5:
-------------------------------------------------
"""
import codecs
import json
import os
import random

import logging
from pydantic import BaseModel
from datetime import datetime

from typing import List, Tuple, Any

from ai_game.common import Board


class PythonObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, Board):
            return obj.board
        if isinstance(obj, BaseModel):
            return obj.dict()
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        return json.JSONEncoder.default(self, obj)


def jdumps(obj):
    return json.dumps(obj, indent=4, ensure_ascii=False, cls=PythonObjectEncoder)


def jdump(obj, f):
    if isinstance(f, str):
        f = codecs.open(f, "w", "utf8")
    with f:
        json.dump(obj, f, indent=4, ensure_ascii=False, cls=PythonObjectEncoder)


def weight_choice(seq: List[Tuple[Any, float]]):
    seq = sorted(seq, key=lambda x: x[1], reverse=True)
    sum_weight = sum(e[1] for e in seq)
    threshold = random.random()
    acc_weight = 0
    for val, weight in seq:
        acc_weight += weight
        if sum_weight * threshold <= acc_weight:
            return val
    return val


# make sure $path exists
def ensure_dir_path(func):
    def wrapper(*args, **kwargs):
        dir_path = kwargs['path']
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return func(*args, **kwargs)

    return wrapper


# make sure the dir path of $path exists
def ensure_file_path(func):
    def wrapper(*args, **kwargs):
        path = kwargs['path']
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return func(*args, **kwargs)

    return wrapper


def read_jsonline(file_path):
    rs = []
    with codecs.open(file_path, "r", "utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rs.append(item)
    return rs


def get_logger(name, level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
