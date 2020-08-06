# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     record
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

from ai_game.common import *
from ai_game.util import PythonObjectEncoder


class Record(BaseModel):
    pass


class ActionRecord(Record):
    state: State
    player_name: str
    action: Action


class ValueRecord(Record):
    state: State
    win_color: Color


class Recorder:
    def __init__(self, record_path):
        self.record_path = record_path

    def do_record(self, record: Record):
        with codecs.open(self.record_path, "a", "utf8") as f:
            f.write(json.dumps(record, ensure_ascii=False, cls=PythonObjectEncoder))
            f.write("\n")
