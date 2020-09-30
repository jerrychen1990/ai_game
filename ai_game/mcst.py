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
import codecs
import copy
import logging
import math
import os
import pickle
from enum import Enum
from random import choice
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

from ai_game.common import State, Action, ProbValueEstimator
from ai_game.model import StateModel
from ai_game.player import Player
from ai_game.record import Recorder, ActionRecord
from ai_game.util import weight_choice, ensure_dir_path

logger = logging.getLogger(__name__)


def ucb(q, p, sum_n, n, c=2):
    return q + c * p * math.sqrt(sum_n) / (1 + n)


def tau_weight(n, tau):
    try:
        return math.pow(n, 1 / tau)
    except OverflowError:
        return math.inf


class PredMode(str, Enum):
    N_PROB = "N_PROB"
    PRIOR_PROB = "PRIOR_PROB"
    RANDOM_PROB = "RANDOM_PROB"


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
        self.train_num = 0

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

    def choose_child_with_n_prob(self, tau=1., strict_mode=False, **kwargs):
        assert not self.is_leaf()
        sum_weight = 0
        item_list = []
        for action, (info, node) in self.child_dict.items():
            choose_weight = tau_weight(info.N, tau)
            sum_weight += choose_weight
            item_list.append(((action, info, node), choose_weight))
        prob_list = [(e[0][0], e[1] / sum_weight if sum_weight else e[1]) for e in item_list]
        if strict_mode:
            item_list = sorted(item_list, key=lambda x: x[1], reverse=True)
            rs_action = item_list[0][0]
        else:
            rs_action = weight_choice(item_list)

        return rs_action, prob_list

    def choose_child_with_prior_prob(self, strict_mode=False, **kwargs):
        assert not self.is_leaf()
        sum_weight = 0
        item_list = []
        for action, (info, node) in self.child_dict.items():
            choose_weight = info.P
            sum_weight += choose_weight
            item_list.append(((action, info, node), choose_weight))
        prob_list = [(e[0][0], e[1] / sum_weight if sum_weight else e[1]) for e in item_list]
        if strict_mode:
            item_list = sorted(item_list, key=lambda x: x[1], reverse=True)
            rs_action = item_list[0][0]
        else:
            rs_action = weight_choice(item_list)

        return rs_action, prob_list

    def choose_child_random(self, **kwargs):
        assert not self.is_leaf()
        item_list = list(self.child_dict.items())
        action, (info, node) = choice(item_list)
        return action, info, node

    def __repr__(self):
        return f"state:\n{str(self.state)}"

    def get_detail_str(self):
        rs = "*" * 32 + "\n"
        rs += f"state:{self.state}\n"


class MCST:
    def __init__(self, game_cls, estimator: ProbValueEstimator = None, rollout_choose_func=None):
        self.node_dict = {}
        self.game_cls = game_cls
        self.estimator: ProbValueEstimator = estimator
        self.estimator_cls = estimator.__class__
        self.root = self.get_node(game_cls.get_init_state())
        self.get_valid_action_func = game_cls.get_valid_action_list
        self.state_trans_func = game_cls.state_trans_func
        self.rollout_choose_func = rollout_choose_func
        self.train_num = 0

    def get_node(self, state: State) -> Node:
        if state not in self.node_dict.keys():
            node = Node(state)
            self.node_dict[state] = node
        return self.node_dict[state]

    def expand_node(self, node: Node):
        valid_action_list: List[Action] = self.get_valid_action_func(node.state)
        prob_list, value = self.estimator.get_prob_value(node.state, valid_action_list)
        assert len(prob_list) == len(valid_action_list)
        child_dict = dict()
        for action, prob in zip(valid_action_list, prob_list):
            next_state = self.state_trans_func(node.state, action)
            next_node = self.get_node(next_state)
            info = Info(P=prob)
            child_dict[action] = (info, next_node)
        node.child_dict = child_dict
        return value

    @ensure_dir_path
    def store(self, path, include_estimator=False):
        mcst_path = os.path.join(path, "mcst.pkl")
        logger.info(f"storing mcst to path:{mcst_path}")
        with codecs.open(mcst_path, "wb") as f:
            pickle.dump(self, f)
        if self.estimator and include_estimator:
            self.estimator.save(path=os.path.join(path, "estimator"))

    @classmethod
    def load(cls, path, include_estimator=False):
        logger.info(f"loading mcst from path:{path}")
        mcst_path = os.path.join(path, "mcst.pkl")
        with open(mcst_path, "rb") as f:
            mcst: MCST = pickle.load(f)
        if include_estimator:
            estimator = mcst.estimator_cls.load(path=os.path.join(path, "estimator"))
            mcst.estimator = estimator
        return mcst

    def train(self, state):
        self.train_num += 1
        node = self.get_node(state)
        node.train_num += 1
        logger.debug(f"training node:\n{node}")
        action_trace = []
        # simulate
        logger.debug("simulating")
        while not node.is_leaf():
            action, info, node = node.choose_child_with_ucb()
            action_trace.append((action, info, node))
        logger.debug(f"after simulating, get node:\n{node}")
        # expand
        logger.debug("expanding")
        v = self.expand_node(node)
        # back propagate
        for action, info, node in action_trace:
            info.N += 1
            info.W += v

    def predict(self, state: State, mode: PredMode = PredMode.N_PROB, **kwargs) -> Action:
        node = self.get_node(state)
        logger.debug(f"predicting node:\n{node}")
        if not node.is_expand():
            self.expand_node(node)
        func_dict = {
            PredMode.N_PROB: node.choose_child_with_n_prob,
            PredMode.PRIOR_PROB: node.choose_child_with_prior_prob,
            PredMode.RANDOM_PROB: node.choose_child_random
        }
        func = func_dict[mode]

        (action, _, _), prob_list = func(**kwargs)
        return action, prob_list

    def __getstate__(self):
        odict = copy.copy(self.__dict__)
        for key in ['estimator']:
            if key in odict.keys():
                del odict[key]
        return odict

    def __setstate__(self, state):
        self.__dict__.update(state)


class TauStrategy:
    MIN_TEMP = 0.00001

    @staticmethod
    def greedy_mode(round):
        return TauStrategy.MIN_TEMP

    @staticmethod
    def train_mode(round, threshold=3):
        return 1 if round < threshold else TauStrategy.MIN_TEMP


class MCSTPlayer(Player):
    def __init__(self, name: str, mcst: MCST,
                 train_num=3, recorder: Recorder = None,
                 mode: PredMode = PredMode.N_PROB,
                 tau_func=TauStrategy.train_mode,
                 strict_mode=False):
        self.name = name
        self.mcst = mcst
        self.train_num = train_num
        self.recorder = recorder
        self.tau_func = tau_func
        self.mode = mode
        self.strict_mode = strict_mode

    def choose_action(self, state: State, action_list: List[Action], round) -> Action:
        if self.train_num:
            logger.info("mcst training")
            for idx in range(self.train_num):
                logger.debug(f"train iteration:{idx}")
                self.mcst.train(state)
        logger.info("mcst predicting...")
        tau = self.tau_func(round)
        action, action_prob_list = self.mcst.predict(state, mode=self.mode, tau=tau, strict_mode=self.strict_mode)
        if self.recorder:
            record: ActionRecord = ActionRecord(state=state, action=action, action_prob_list=action_prob_list,
                                                player_name=self.name)
            self.recorder.do_record(record)

        return action
