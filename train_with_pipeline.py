# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train_with_pipeline
   Description :
   Author :       chenhao
   date：          2020/8/12
-------------------------------------------------
   Change Activity:
                   2020/8/12:
-------------------------------------------------
"""
import os

from ai_game.minmax import MinMaxTree
from ai_game.model import get_state_model_of_game
from ai_game.tic_tac_toe import TicTacToe
from ai_game.pipeline import train


def train_with_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    build_kwargs = {
        "embedding_dim": 32,
        "filter_list": [],
        "dense_list": [32, 32]
    }
    player_kwargs = {
        "train_num": 3
    }
    train_kwargs = {
        "epochs": 5,
        "batch_size": 8
    }
    game_cls = TicTacToe
    estimator = get_state_model_of_game(game_cls=game_cls, build_kwargs=build_kwargs)

    train(game_cls=game_cls, job_name="test_job", estimator=estimator, iteration_num=2, game_num=2,
          build_kwargs=build_kwargs,
          player_kwargs=player_kwargs, train_kwargs=train_kwargs)


def train_with_minmax():
    player_kwargs = {
        "train_num": 32,
        "strict_mode": False
    }
    train_kwargs = {

    }
    game_cls = TicTacToe
    estimator = MinMaxTree.load(path="/Users/haochen/workspace/ai_game/ckpt/minmax-tictactoe")

    train(game_cls=game_cls, job_name="test_minmax_job", estimator=estimator, iteration_num=1, game_num=100,
          player_kwargs=player_kwargs, train_kwargs=train_kwargs)


if __name__ == '__main__':
    # train_with_model()
    train_with_minmax()
