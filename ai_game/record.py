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
from pydantic import BaseModel
from typing import List, Tuple

from ai_game.common import State, PutPieceAction, Color
from ai_game.util import PythonObjectEncoder, ensure_file_path


class Record(BaseModel):
    pass


class ActionRecord(Record):
    state: State
    player_name: str
    action: PutPieceAction
    action_prob_list: List[Tuple[PutPieceAction, float]]
    win_color: Color = None


@ensure_file_path
def dump_record(path, record_list):
    with codecs.open(path, "w", "utf8") as f:
        for record in record_list:
            f.write(json.dumps(record, ensure_ascii=False, cls=PythonObjectEncoder))
            f.write("\n")


class Recorder:
    def __init__(self, record_path):
        self.record_path = record_path
        self.record_list: List[ActionRecord] = []

    def do_record(self, record: ActionRecord):
        self.record_list.append(record)

    def update_win_color(self, win_color):
        for record in self.record_list:
            record.win_color = win_color

    def dump_record(self):
        dump_record(path=self.record_path, record_list=self.record_list)
