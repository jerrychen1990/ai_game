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
from ai_game.game import *
from ai_game.mcst import *
from ai_game.model import get_state_model_of_game
from ai_game.record import Recorder

logger = logging.getLogger(__name__)
cur_path = os.path.abspath(os.path.dirname(__file__))


def train(game_cls, job_name, estimator, iteration_num, game_num, player_kwargs, train_kwargs):
    logger.info("job start")

    logger.info(f"initializing state model")
    for iteration in range(iteration_num):
        logger.info(f"iteration:{iteration_num}")
        logger.info(f"initializing mcst")
        mcst = MCST(game_cls, estimator=estimator)
        logger.info(f"initializing recorder")
        recorder = Recorder(record_path=os.path.join(cur_path, f"../record/{job_name}/{iteration}.jsonl"))
        player_kwargs["recorder"] = recorder
        player_list = [MCSTPlayer(name="mcst_player1", mcst=mcst, **player_kwargs),
                       MCSTPlayer(name="mcst_player2", mcst=mcst, **player_kwargs)]
        logger.info("starting competition")
        for game_idx in range(game_num):
            logger.info(f"game_idx:{game_idx}")
            game = game_cls(player_list=player_list, recorder=recorder, level=logging.DEBUG)
            game.start()
        logger.info(f"dumping records")
        recorder.dump_record()
        logger.info("training model")
        train_data = recorder.record_list
        estimator.train(data=train_data, **train_kwargs)
        logger.info("saving mcst")
        store_path = os.path.join(cur_path, f"../ckpt/{job_name}/{iteration}")
        mcst.store(path=store_path, include_estimator=True)
    # logger.info("saving estimator")
    # estimator.save(path=store_path)

    logger.info("job done")


