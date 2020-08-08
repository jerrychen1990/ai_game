# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_model
   Description :
   Author :       chenhao
   date：          2020/8/7
-------------------------------------------------
   Change Activity:
                   2020/8/7:
-------------------------------------------------
"""
import unittest

from ai_game.model import *
from ai_game.tic_tac_toe import TicTacToe

logger = logging.getLogger(__name__)

build_kwargs = {"embedding_dim": 20,
                "filter_list": []}


class TestModel(unittest.TestCase):
    def test_get_state_model_of_game(self):
        model = get_state_model_of_game(TicTacToe, build_kwargs=build_kwargs)
        return model

    def test_predict(self):
        model = get_state_model_of_game(TicTacToe, build_kwargs=build_kwargs)
        state = TicTacToe.get_init_state()
        validate_action_list = TicTacToe.get_valid_action_list(state)
        action_list, value = model.predict(state, validate_action_list)
        print(action_list)
        print(value)

    def test_train(self):
        model = get_state_model_of_game(TicTacToe, build_kwargs=build_kwargs)
        data = [{"state": {"board": {"row_num": 3, "col_num": 3,
                                     "board": [[{"color": "X"}, {"color": "O"}, {"color": "X"}],
                                               [{"color": "O"}, {"color": "X"}, {"color": "O"}],
                                               [{"color": "X"}, None, None]]}, "color": "X"},
                 "player_name": "mcst_player1", "action": {"row": 0, "col": 0, "piece": {"color": "X"}},
                 "action_prob_list": [[{"row": 0, "col": 0, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 0, "col": 1, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 0, "col": 2, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 1, "col": 0, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 1, "col": 1, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 1, "col": 2, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 2, "col": 0, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 2, "col": 1, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 2, "col": 2, "piece": {"color": "X"}}, 0.0]], "win_color": "X"},
                {"state": {"board": {"row_num": 3, "col_num": 3,
                                     "board": [[{"color": "X"}, {"color": "O"}, {"color": "X"}],
                                               [{"color": "O"}, {"color": "X"}, {"color": "O"}],
                                               [{"color": "X"}, None, None]]}, "color": "O"},
                 "player_name": "mcst_player2", "action": {"row": 0, "col": 1, "piece": {"color": "O"}},
                 "action_prob_list": [[{"row": 0, "col": 1, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 0, "col": 2, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 1, "col": 0, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 1, "col": 1, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 1, "col": 2, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 2, "col": 0, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 2, "col": 1, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 2, "col": 2, "piece": {"color": "O"}}, 0.0]], "win_color": "X"},
                {"state": {"board": {"row_num": 3, "col_num": 3,
                                     "board": [[{"color": "X"}, {"color": "O"}, {"color": "X"}],
                                               [{"color": "O"}, {"color": "X"}, {"color": "O"}],
                                               [{"color": "X"}, None, None]]}, "color": "X"},
                 "player_name": "mcst_player1", "action": {"row": 0, "col": 2, "piece": {"color": "X"}},
                 "action_prob_list": [[{"row": 0, "col": 2, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 1, "col": 0, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 1, "col": 1, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 1, "col": 2, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 2, "col": 0, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 2, "col": 1, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 2, "col": 2, "piece": {"color": "X"}}, 0.0]], "win_color": "X"},
                {"state": {"board": {"row_num": 3, "col_num": 3,
                                     "board": [[{"color": "X"}, {"color": "O"}, {"color": "X"}],
                                               [{"color": "O"}, {"color": "X"}, {"color": "O"}],
                                               [{"color": "X"}, None, None]]}, "color": "O"},
                 "player_name": "mcst_player2", "action": {"row": 1, "col": 0, "piece": {"color": "O"}},
                 "action_prob_list": [[{"row": 1, "col": 0, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 1, "col": 1, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 1, "col": 2, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 2, "col": 0, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 2, "col": 1, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 2, "col": 2, "piece": {"color": "O"}}, 0.0]], "win_color": "X"},
                {"state": {"board": {"row_num": 3, "col_num": 3,
                                     "board": [[{"color": "X"}, {"color": "O"}, {"color": "X"}],
                                               [{"color": "O"}, {"color": "X"}, {"color": "O"}],
                                               [{"color": "X"}, None, None]]}, "color": "O"},
                 "player_name": "mcst_player2", "action": {"row": 1, "col": 2, "piece": {"color": "O"}},
                 "action_prob_list": [[{"row": 1, "col": 2, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 2, "col": 0, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 2, "col": 1, "piece": {"color": "O"}}, 0.0],
                                      [{"row": 2, "col": 2, "piece": {"color": "O"}}, 0.0]], "win_color": "X"},
                {"state": {"board": {"row_num": 3, "col_num": 3,
                                     "board": [[{"color": "X"}, {"color": "O"}, {"color": "X"}],
                                               [{"color": "O"}, {"color": "X"}, {"color": "O"}],
                                               [{"color": "X"}, None, None]]}, "color": "X"},
                 "player_name": "mcst_player1", "action": {"row": 2, "col": 0, "piece": {"color": "X"}},
                 "action_prob_list": [[{"row": 2, "col": 0, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 2, "col": 1, "piece": {"color": "X"}}, 0.0],
                                      [{"row": 2, "col": 2, "piece": {"color": "X"}}, 0.0]], "win_color": "X"}]

        record_list = [ActionRecord(**e) for e in data]

        model.train(data=record_list, epochs=1, batch_size=1)
