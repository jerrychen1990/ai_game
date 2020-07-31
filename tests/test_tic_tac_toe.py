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
from ai_game.tic_tac_toe import *
from ai_game.mcst import MCST, MCSTPlayer
from ai_game.player import RandomPlayer, HumanPlayer


class TestTicTacToe(unittest.TestCase):
    def test_tic_tac_toe_with_random(self):
        game = TicTacToe([RandomPlayer("random_player1"),
                          RandomPlayer("random_player2")])
        game.start()

    def test_tic_tac_toe_with_human(self):
        game = TicTacToe([RandomPlayer("random_player1"),
                          HumanPlayer("human_player2")])
        game.start()

    def test_tic_tac_toe_with_mcst(self):
        mcst = MCST(game_cls=TicTacToe, rollout_choose_func=Strategy.random_choose)

        game = TicTacToe([MCSTPlayer("mcst_player1", mcst, 3),
                          MCSTPlayer("mcst_player2", mcst, 3)])
        game.start()

    def test_tic_tac_toe_with_mcst_vs_human(self):
        mcst_path = "../ckpt/mcst-TicTacToe-100-5.pkl"
        mcst = MCST.load(mcst_path)

        game = TicTacToe([MCSTPlayer("mcst_player1", mcst, train_num=3, is_train=False),
                          HumanPlayer("human_player2")])
        game.start()
