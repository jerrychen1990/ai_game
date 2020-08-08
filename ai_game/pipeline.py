# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     pipeline
   Description :
   Author :       chenhao
   date：          2020/8/7
-------------------------------------------------
   Change Activity:
                   2020/8/7:
-------------------------------------------------
"""
import logging
import os
from ai_game.mcst import *
from ai_game.game import *
from ai_game.model import get_state_model_of_game
from ai_game.record import Recorder
from ai_game.tic_tac_toe import TicTacToe

logger = logging.getLogger(__name__)
cur_path = os.path.abspath(os.path.dirname(__file__))


def train(game_cls, job_name, iteration_num, round_num, build_kwargs, player_kwargs, train_kwargs):
    logger.info("job start")

    logger.info(f"initializing state model")
    model = get_state_model_of_game(game_cls=game_cls, build_kwargs=build_kwargs)
    for iteration in range(iteration_num):
        logger.info(f"iteration:{iteration_num}")
        logger.info(f"initializing mcst")
        mcst = MCST(game_cls, model=model)
        logger.info(f"initializing recorder")
        recorder = Recorder(record_path=os.path.join(cur_path, f"../record/{job_name}/{iteration}.jsonl"))
        player_kwargs["recorder"] = recorder
        player_list = [MCSTPlayer(name="mcst_player1", mcst=mcst, **player_kwargs),
                       MCSTPlayer(name="mcst_player2", mcst=mcst, **player_kwargs)]
        logger.info("starting competition")
        for round in range(round_num):
            logger.info(f"round:{round}")
            game = game_cls(player_list=player_list, recorder=recorder)
            game.start()
        logger.info(f"dumping records")
        recorder.dump_record()
        logger.info("training model")
        train_data = recorder.record_list
        model.train(data=train_data, **train_kwargs)

    logger.info("job done")


if __name__ == '__main__':
    build_kwargs = {
        "embedding_dim": 20,
        "filter_list": []
    }
    player_kwargs = {
        "train_num": 1,
        "is_train": True
    }
    train_kwargs = {
        "epochs": 2,
        "batch_size": 1
    }

    train(game_cls=TicTacToe, job_name="test_job", iteration_num=1, round_num=1, build_kwargs=build_kwargs,
          player_kwargs=player_kwargs, train_kwargs=train_kwargs)
