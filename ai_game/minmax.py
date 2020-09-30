# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     minmax
   Description :
   Author :       chenhao
   date：          2020/8/13
-------------------------------------------------
   Change Activity:
                   2020/8/13:
-------------------------------------------------
"""
import logging
import math
import os
import pickle
from typing import Dict, List, Tuple

from ai_game.common import State, Action, ProbValueEstimator, Trainable
from ai_game.util import ensure_file_path, ensure_dir_path

logger = logging.getLogger(__name__)


class Node:

    def __init__(self, state):
        self.state: State = state
        self.child_dict: Dict[Action:Node] = None
        self.value = None

    def __repr__(self):
        return f"state:\n{str(self.state)}\nvalue:{self.value}"


class MinMaxTree(ProbValueEstimator, Trainable):
    def __init__(self, game_cls):
        self.node_dict = {}
        self.game_cls = game_cls
        self.root = self.get_node(game_cls.get_init_state())
        self.get_valid_action_func = game_cls.get_valid_action_list
        self.state_trans_func = game_cls.state_trans_func
        self.get_win_color_func = game_cls.get_win_color
        self.parsed_node_num = 0

    def get_node(self, state: State):
        if state not in self.node_dict.keys():
            node = Node(state)
            self.node_dict[state] = node
        return self.node_dict[state]

    def expand(self, node: Node):
        valid_action_list = self.get_valid_action_func(node.state)
        node.child_dict = {}
        for action in valid_action_list:
            next_state = self.state_trans_func(node.state, action)
            next_node = self.get_node(next_state)
            node.child_dict[action] = next_node

    def get_value(self, node: Node):
        if node.value is not None:
            return node.value
        # logger.info(f"getting value of state:{node.state}")
        win_color = self.get_win_color_func(node.state)
        if win_color is not None:
            value = 1 if node.state.color == win_color else -1
        else:
            if node.child_dict is None:
                self.expand(node)
            if len(node.child_dict) == 0:
                value = 0
            else:
                value_list = [self.get_value(n) for n in node.child_dict.values()]
                value = -min(value_list)
        # logger.info(f"state:{node.state}'s value is {value}")
        node.value = value
        self.parsed_node_num += 1
        logger.info(f"current process:{self.parsed_node_num}/{math.pow(3, 9)}")
        return value

    def eval_state(self, state: State) -> float:
        node = self.get_node(state)
        return self.get_value(node)

    def get_prob(self, state: State, action_list: List[Action], base=10) -> List[float]:
        node = self.get_node(state)
        self.get_value(node)
        if node.child_dict is not None:
            value_list = [self.get_value(node.child_dict[action]) for action in action_list]
        else:
            value_list = [1 for _ in action_list]
        prob_list = [math.pow(base, -v) for v in value_list]
        sum_prob = sum(prob_list)
        prob_list = [e / sum_prob for e in prob_list]
        return prob_list

    def train(self, data, **kwargs):
        pass

    @ensure_dir_path
    def save(self, path):
        pickle_path = os.path.join(path, "minmax.pkl")
        logger.info(f"saving minmax tree to path:{pickle_path}")
        pickle.dump(self, open(pickle_path, "wb"))

    @classmethod
    @ensure_dir_path
    def load(cls, path):
        pickle_path = os.path.join(path, "minmax.pkl")
        logger.info(f"loading minmax tree from path:{pickle_path}")
        instance = pickle.load(open(pickle_path, "rb"))
        return instance
