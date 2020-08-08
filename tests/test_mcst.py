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
from ai_game.model import get_state_model_of_game
from ai_game.tic_tac_toe import TicTacToe

logger = logging.getLogger(__name__)


class TestMCST(unittest.TestCase):
    def test_train(self):
        build_kwargs = {"embedding_dim": 20,
                        "filter_list": []}
        model = get_state_model_of_game(TicTacToe, build_kwargs=build_kwargs)

        train_num = 10
        mcst = MCST(game_cls=TicTacToe, model=model)
        state = TicTacToe.get_init_state()

        for idx in range(train_num):
            mcst.train(state)

    def test_predict(self):
        build_kwargs = {"embedding_dim": 20,
                        "filter_list": []}
        model = get_state_model_of_game(TicTacToe, build_kwargs=build_kwargs)

        mcst = MCST(game_cls=TicTacToe, model=model)
        state = TicTacToe.get_init_state()
        action = mcst.predict(state)
        print(action)

    @unittest.skip("changed mcst")
    def test_load(self):
        mcst = MCST.load(load_path="../ckpt/mcst-TicTacToe-10-10.pkl")
        mcst
