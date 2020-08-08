# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_record
   Description :
   Author :       chenhao
   date：          2020/8/8
-------------------------------------------------
   Change Activity:
                   2020/8/8:
-------------------------------------------------
"""

import unittest
from kg_schema.utils import show_pydantic_obj
from ai_game.record import *


class TestRecord(unittest.TestCase):
    def test_record(self):
        record_dict = {"state": {"board": {"row_num": 3, "col_num": 3,
                                           "board": [[{"color": "X"}, {"color": "O"}, {"color": "X"}],
                                                     [{"color": "O"}, {"color": "X"}, {"color": "O"}],
                                                     [{"color": "X"}, None, None]]}, "color": "X"},
                       "player_name": "mcst_player1", "action": {"row": 0, "col": 0, "piece": {"color": "X"}},
                       "action_prob_list": [[{"row": 0, "col": 0, "piece": {"color": "X"}}, 0.0],
                                            [{"row": 0, "col": 1, "piece": {"color": "X"}}, 0.0],
                                            [{"row": 0, "col": 2, "piece": {"color": "X"}}, 0.0],
                                            [{"row": 1, "col": 0, "piece": {"color": "X"}}, 0.0],
                                            [{"row": 1, "col": 1, "piece": {"color": "X"}}, 0.0],
                                            [{"row": 1, "col": 2, "piece": {"color": "X"}}, 0.0],
                                            [{"row": 2, "col": 0, "piece": {"color": "X"}}, 0.0],
                                            [{"row": 2, "col": 1, "piece": {"color": "X"}}, 0.0],
                                            [{"row": 2, "col": 2, "piece": {"color": "X"}}, 0.0]], "win_color": "X"}
        record = ActionRecord(**record_dict)
        show_pydantic_obj(record)
