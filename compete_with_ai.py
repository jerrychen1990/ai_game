# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     compete_with_ai
   Description :
   Author :       chenhao
   date：          2020/8/8
-------------------------------------------------
   Change Activity:
                   2020/8/8:
-------------------------------------------------
"""
import os
from ai_game.competition import Competition
from ai_game.mcst import MCST, MCSTPlayer, PredMode
from ai_game.tic_tac_toe import TicTacToe
from ai_game.player import HumanPlayer

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    match_num = 5

    mcst_path = "/Users/haochen/workspace/ai_game/ai_game/../ckpt/test_job/9"
    mcst_path = "/Users/haochen/workspace/ai_game/ai_game/../ckpt/test_minmax_job/0"

    mcst = MCST.load(mcst_path, include_estimator=True)

    game_cls = TicTacToe

    player_list = [MCSTPlayer("mcst_player1", mcst, train_num=0, mode=PredMode.N_PROB, strict_mode=True),
                   HumanPlayer("human_player2")]
    competition = Competition(game_cls, player_list, match_num)
    competition.start()
