# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_game
   Description :
   Author :       chenhao
   date：          2020/7/29
-------------------------------------------------
   Change Activity:
                   2020/7/29:
-------------------------------------------------
"""
import unittest

from ai_game.common import Strategy
from ai_game.record import Recorder
from ai_game.tic_tac_toe import *
from ai_game.mcst import MCST, MCSTPlayer
from ai_game.player import RandomPlayer, HumanPlayer


class TestTicTacToe(unittest.TestCase):
    def test_tic_tac_toe_with_random(self):
        game = TicTacToe([RandomPlayer("random_player1"),
                          RandomPlayer("random_player2")])
        game.start()

    @unittest.skip("changed mcst")
    def test_tic_tac_toe_with_mcst(self):
        mcst = MCST(game_cls=TicTacToe, rollout_choose_func=Strategy.random_choose)
        recorder = Recorder(record_path="../record/test_record.jsonl")

        game = TicTacToe([MCSTPlayer("mcst_player1", mcst, 3),
                          MCSTPlayer("mcst_player2", mcst, 3)], recorder)
        game.start()
