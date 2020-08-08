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
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Type

from ai_game.constant import *
from ai_game.common import State, Action
from ai_game.record import Recorder, ActionRecord
from ai_game.util import weight_choice
from ai_game.game import Game
from ai_game.model import StateModel
from ai_game.player import Player

logger = logging.getLogger(__name__)


def ucb(q, p, sum_n, n, c=2):
    return q + c * p * math.sqrt(sum_n) / (1 + n)


class Info(BaseModel):
    N: int = Field(0, description="访问次数")
    P: float = Field(0, description="先验概率")
    W: float = Field(0, description="行动价值的总和")

    @property
    def Q(self):
        return self.W / self.N if self.N else 0.


class Node:

    def __init__(self, state):
        self.state: State = state
        self.child_dict: Dict[Action:Tuple[Info, Node]] = None

    def is_expand(self):
        return self.child_dict is not None

    def is_leaf(self):
        return not self.child_dict

    def is_terminate(self):
        return self.is_expand() and self.is_leaf()

    def choose_child_with_ucb(self, c=2):
        assert not self.is_leaf()
        item_list = []
        sum_n = sum(info.N for info, _ in self.child_dict.values())
        for action, (info, node) in self.child_dict.items():
            ucb_value = ucb(info.Q, info.P, sum_n, info.N, c)
            item_list.append((action, info, node, ucb_value))
        item_list = sorted(item_list, key=lambda x: x[-1], reverse=True)
        return item_list[0][:3]

    def choose_child_with_n_prob(self, tau=1., return_prob=False):
        assert not self.is_leaf()
        sum_weight = 0
        item_list = []
        for action, (info, node) in self.child_dict.items():
            choose_weight = math.pow(info.N, tau)
            sum_weight += choose_weight
            item_list.append(((action, info, node), choose_weight))
        prob_list = [(e[0][0], e[1] / sum_weight if sum_weight else e[1]) for e in item_list]
        rs_action = weight_choice(item_list)

        return rs_action, prob_list if return_prob else rs_action

    def choose_child_random(self):
        assert not self.is_leaf()
        item_list = list(self.child_dict.items())
        action, (info, node) = choice(item_list)
        return action, info, node

    def __repr__(self):
        return f"state:\n{str(self.state)}"


class MCST:
    def __init__(self, game_cls, model: StateModel = None, rollout_choose_func=None):
        self.node_dict = {}
        self.game_cls = game_cls
        self.root = self.get_node(game_cls.get_init_state())
        self.get_valid_action_func = game_cls.get_valid_action_list
        self.state_trans_func = game_cls.state_trans_func
        self.model = model
        self.rollout_choose_func = rollout_choose_func

    def get_node(self, state: State):
        if state not in self.node_dict.keys():
            node = Node(state)
            self.node_dict[state] = node
        return self.node_dict[state]

    def expand_node(self, node: Node):
        valid_action_list: List[Action] = self.get_valid_action_func(node.state)
        prob_list, value = self.model.predict(node.state, valid_action_list)
        assert len(prob_list) == len(valid_action_list)
        child_dict = dict()
        for action, prob in zip(valid_action_list, prob_list):
            next_state = self.state_trans_func(node.state, action)
            next_node = self.get_node(next_state)
            info = Info(P=prob)
            child_dict[action] = (info, next_node)
        node.child_dict = child_dict
        return value

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
        action_trace = []
        # simulate
        logger.info("simulating")
        while not node.is_leaf():
            action, info, node = node.choose_child_with_ucb()
            action_trace.append((action, info, node))
        logger.debug(f"after simulating, get node:\n{node}")
        # expand
        logger.info("expanding")
        v = self.expand_node(node)
        # back propagate
        for action, info, node in action_trace:
            info.N += 1
            info.W += v

    def predict(self, state: State, tau=1.) -> Action:
        node = self.get_node(state)
        logger.debug(f"predicting node:\n{node}")
        if not node.is_expand():
            self.expand_node(node)
        (action, _, _), prob_list = node.choose_child_with_n_prob(tau, return_prob=True, )
        return action, prob_list


class MCSTPlayer(Player):
    def __init__(self, name: str, mcst: MCST, train_num=3, is_train=True, recorder: Recorder = None):
        self.name = name
        self.mcst = mcst
        self.train_num = train_num
        self.is_train = is_train
        self.recorder = recorder

    def choose_action(self, state: State, action_list: List[Action]) -> Action:
        if self.is_train:
            logger.info("mcst training")
            for idx in range(self.train_num):
                logger.info(f"train iteration:{idx}")
                self.mcst.train(state)
        logger.info("mcst predicting...")
        action, action_prob_list = self.mcst.predict(state)
        if self.recorder:
            record: ActionRecord = ActionRecord(state=state, action=action, action_prob_list=action_prob_list,
                                                player_name=self.name)
            self.recorder.do_record(record)

        return action
