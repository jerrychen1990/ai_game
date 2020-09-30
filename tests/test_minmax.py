# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_minmax
   Description :
   Author :       chenhao
   date：          2020/8/13
-------------------------------------------------
   Change Activity:
                   2020/8/13:
-------------------------------------------------
"""

import logging
import unittest
import os
from ai_game.minmax import *
from ai_game.tic_tac_toe import TicTacToe

cur_path = os.path.abspath(os.path.dirname(__file__))

logger = logging.getLogger(__name__)


class TestMinMax(unittest.TestCase):
    def test_minmax_tree(self):
        minmax_tree = MinMaxTree(TicTacToe)
        value = minmax_tree.get_value(minmax_tree.root)
        logger.info(value)
        minmax_tree.save(path=os.path.join(cur_path, "../ckpt/minmax/tictactoe.pkl"))

    def test_explore_minmax_tree(self):
        minmax_tree = MinMaxTree.load(path="../ckpt/minmax/tictactoe.pkl")
        print("load finish")
