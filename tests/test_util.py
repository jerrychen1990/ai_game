# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_util
   Description :
   Author :       chenhao
   date：          2020/8/5
-------------------------------------------------
   Change Activity:
                   2020/8/5:
-------------------------------------------------
"""
import unittest
import logging
import json

from ai_game.util import jdumps, PythonObjectEncoder
from ai_game.tic_tac_toe import TicTacToe

logger = logging.getLogger(__name__)


class TestUtil(unittest.TestCase):
    def test_jdumps(self):
        state = TicTacToe.get_init_state()
        logger.info(jdumps(state))
