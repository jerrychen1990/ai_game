# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     compare_models
   Description :
   Author :       chenhao
   date：          2020/8/12
-------------------------------------------------
   Change Activity:
                   2020/8/12:
-------------------------------------------------
"""
from ai_game.competition import Competition
from ai_game.tic_tac_toe import TicTacToe
from ai_game.mcst import MCST, MCSTPlayer, TauStrategy
from ai_game.model import StateModel

if __name__ == '__main__':
    train_num = 3
    match_num = 100
    game_cls = TicTacToe

    state_model1 = StateModel.load(path="/Users/haochen/workspace/ai_game/ckpt/test_job/0")
    mcst1 = MCST(game_cls=game_cls, model=state_model1)
    mcst_player1 = MCSTPlayer("mcst_player1", mcst1, train_num, tau_func=TauStrategy.greedy_mode)

    state_model2 = StateModel.load(path="/Users/haochen/workspace/ai_game/ckpt/test_job/9")
    mcst2 = MCST(game_cls=game_cls, model=state_model2)
    mcst_player2 = MCSTPlayer("mcst_player2", mcst2, train_num, tau_func=TauStrategy.greedy_mode)

    player_list = [mcst_player1, mcst_player2]
    competition = Competition(game_cls, player_list, match_num)
    competition.start()
