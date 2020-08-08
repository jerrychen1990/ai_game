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

from ai_game.common import Strategy
from ai_game.competition import Competition
from ai_game.mcst import MCST, MCSTPlayer
from ai_game.player import RandomPlayer, HumanPlayer
from ai_game.tic_tac_toe import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)s] [%(levelname)s]: %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')


class TestCompetition(unittest.TestCase):
    def test_competition(self):
        player_list = [RandomPlayer("random_player1"),
                       RandomPlayer("random_player2")]
        game_cls = TicTacToe
        match_num = 10
        competition = Competition(game_cls, player_list, match_num)
        competition.start()

    @unittest.skip("changed mcst")
    def test_mcst_competition(self):
        train_num = 5
        match_num = 100

        game_cls = TicTacToe
        mcst = MCST(game_cls=game_cls, rollout_choose_func=Strategy.random_choose)

        player_list = [MCSTPlayer("mcst_player1", mcst, train_num),
                       MCSTPlayer("mcst_player2", mcst, train_num)]
        competition = Competition(game_cls, player_list, match_num)
        competition.start()
        mcst.store(f"../ckpt/mcst-{TicTacToe.__name__}-{match_num}-{train_num}.pkl")
