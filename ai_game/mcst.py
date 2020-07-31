# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     mcst
   Description :
   Author :       chenhao
   date：          2020/7/30
-------------------------------------------------
   Change Activity:
                   2020/7/30:
-------------------------------------------------
"""
import logging
import math
import pickle
from random import choice
from typing import Dict, List

from ai_game.constant import *
from ai_game.common import State, Action
from ai_game.game import Game
from ai_game.player import Player

logger = logging.getLogger(__name__)


def ucb(v, parent_n, n, c=2, epsilon=1e-5):
    return v + c * math.sqrt(math.log(parent_n) / (n + epsilon))


class Node:

    def __init__(self, state):
        self.state: State = state
        self.visit_num: int = 0
        self.total_value: float = 0
        self.child_dict: Dict[Action:Node] = None

    def is_expand(self):
        return self.child_dict is not None

    def is_leaf(self):
        return not self.child_dict

    def is_terminate(self):
        return self.is_expand() and self.is_leaf()

    def add_value(self, val: float):
        self.visit_num += 1
        self.total_value += val

    def get_avg_value(self):
        return self.total_value / (self.visit_num + EPSILON)

    def get_node_value(self, node, positive_color):
        avg_value = node.get_avg_value()
        return avg_value if positive_color == self.state.color else 1 - avg_value

    def choose_child_with_ucb(self, positive_color, c=2):
        assert not self.is_leaf()
        item_list = []
        for action, node in self.child_dict.items():
            node_value = self.get_node_value(node, positive_color)
            ucb_value = ucb(node_value, self.visit_num, node.visit_num, c=c, epsilon=EPSILON)
            item_list.append((action, node, ucb_value))
        item_list = sorted(item_list, key=lambda x: x[2], reverse=True)
        return item_list[0][:2]

    def choose_child_greedy(self, positive_color):
        assert not self.is_leaf()
        item_list = list(self.child_dict.items())
        item_list = sorted(item_list, key=lambda x: self.get_node_value(x[1], positive_color), reverse=True)
        return item_list[0]

    def choose_child_random(self):
        assert not self.is_leaf()
        item_list = list(self.child_dict.items())
        return choice(item_list)

    def __repr__(self):
        return f"state:\n{str(self.state)}\nvisit_num:{self.visit_num},total_value:{self.total_value},avg_value:{self.get_avg_value():2.3} "


class MCST:
    def __init__(self, game_cls: type(Game), rollout_choose_func):
        self.node_dict: Dict[State, Node] = dict()
        self.root = self.get_node(game_cls.get_init_state())
        self.positive_color = game_cls.COLOR_LIST[0]
        self.game_cls = game_cls
        self.get_valid_action_func = self.game_cls.get_valid_action_list
        self.state_trans_func = self.game_cls.state_trans_func
        self.judge_end_func = self.game_cls.is_terminate
        self.get_win_color_func = self.game_cls.get_win_color
        self.rollout_choose_func = rollout_choose_func

    def get_node(self, state: State):
        if state not in self.node_dict.keys():
            node = Node(state)
            self.node_dict[state] = node
        return self.node_dict[state]

    def expand_node(self, node):
        valid_action_list: List[Action] = self.get_valid_action_func(node.state)
        child_dict = dict()
        for action in valid_action_list:
            next_state = self.state_trans_func(node.state, action)
            next_node = self.get_node(next_state)
            child_dict[action] = next_node
        node.child_dict = child_dict

    def store(self, store_path):
        logger.info(f"storing mcst to path:{store_path}")

        with open(store_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_path):
        with open(load_path, "rb") as f:
            mcst = pickle.load(f)
            return mcst

    def train(self, state):
        node = self.get_node(state)
        logger.debug(f"training node:\n{node}")
        node_trace = [node]
        # simulate
        logger.info("simulating")
        while not node.is_leaf():
            _, node = node.choose_child_with_ucb(self.positive_color)
            node_trace.append(node)
        logger.debug(f"after simulating, get node:\n{node}")
        # expand
        if not node.is_expand():
            logger.info("expanding")
            self.expand_node(node)

        # rollout
        logger.info("rollouting")
        cur_state = node.state
        while not self.judge_end_func(cur_state):
            valid_action_list = self.get_valid_action_func(cur_state)
            action = self.rollout_choose_func(cur_state, valid_action_list)
            cur_state = self.state_trans_func(cur_state, action)
        win_color = self.get_win_color_func(cur_state)
        if win_color:
            value = 1. if win_color == self.positive_color else 0.
        else:
            value = .5

        # back propagate
        logger.info(f"back propagating with value:{value}")
        for node in node_trace:
            node.add_value(value)

    def predict(self, state: State) -> Action:
        node = self.get_node(state)
        logger.debug(f"predicting node:\n{node}")
        if not node.is_expand():
            self.expand_node(node)
        action, _ = node.choose_child_greedy(self.positive_color)
        return action


class MCSTPlayer(Player):
    def __init__(self, name: str, mcst: MCST, train_num=3, is_train=True):
        self.name = name
        self.mcst = mcst
        self.train_num = train_num
        self.is_train = is_train

    def choose_action(self, state: State, action_list: List[Action]) -> Action:
        if self.is_train:
            logger.info("mcst training")
            for idx in range(self.train_num):
                logger.info(f"train iteration:{idx}")
                self.mcst.train(state)
        logger.info("mcst predicting...")
        action = self.mcst.predict(state)
        return action
