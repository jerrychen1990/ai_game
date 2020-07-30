# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_competition
   Description :
   Author :       chenhao
   date：          2020/7/30
-------------------------------------------------
   Change Activity:
                   2020/7/30:
-------------------------------------------------
"""
import unittest
from ai_game.competition import Competition
from ai_game.tic_tac_toe import *
from ai_game.player import RandomPlayer, HumanPlayer


class TestCompetition(unittest.TestCase):
    def test_competition(self):
        player_list = [RandomPlayer("random_player1"),
                       RandomPlayer("random_player2")]
        game_cls = TicTacToe
        match_num = 10
        competition = Competition(game_cls, player_list, match_num)
        competition.start()
