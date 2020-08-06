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
import json
from pydantic import BaseModel
from datetime import datetime

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
            return obj.strftime("%Y-%m-%dT%H:%M:%S")
        return json.JSONEncoder.default(self, obj)


def jdumps(obj):
    return json.dumps(obj, indent=4, ensure_ascii=False, cls=PythonObjectEncoder)
