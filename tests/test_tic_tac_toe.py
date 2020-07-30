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
from ai_game.tic_tac_toe import *
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
