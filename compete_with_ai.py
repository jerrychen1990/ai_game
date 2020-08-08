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
from ai_game.competition import Competition
from ai_game.mcst import MCST, MCSTPlayer
from ai_game.tic_tac_toe import TicTacToe
from ai_game.player import HumanPlayer

if __name__ == '__main__':
    match_num = 5

    mcst_path = "../ckpt/mcst-TicTacToe-100-5.pkl"
    mcst = MCST.load(mcst_path)

    game_cls = TicTacToe

    player_list = [MCSTPlayer("mcst_player1", mcst, is_train=False),
                   HumanPlayer("human_player2")]
    competition = Competition(game_cls, player_list, match_num)
    competition.start()
