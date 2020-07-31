# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_mcst
   Description :
   Author :       chenhao
   date：          2020/7/30
-------------------------------------------------
   Change Activity:
                   2020/7/30:
-------------------------------------------------
"""
import unittest

from ai_game.common import Color, Strategy, Board
from ai_game.mcst import *
from ai_game.tic_tac_toe import TicTacToe

logger = logging.getLogger(__name__)


class TestMCST(unittest.TestCase):
    def test_train(self):
        train_num = 10
        mcst = MCST(Color.X, game_cls=TicTacToe, rollout_choose_func=Strategy.random_choose)
        board = Board(row_num=3)
        root_state = State(board=board, color=Color.X)
        for idx in range(train_num):
            mcst.train(root_state)

    def test_load(self):
        mcst = MCST.load(load_path="../ckpt/mcst-TicTacToe-10-10.pkl")
        mcst
