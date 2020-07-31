# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     game
   Description :
   Author :       chenhao
   date：          2020/7/29
-------------------------------------------------
   Change Activity:
                   2020/7/29:
-------------------------------------------------
"""
import logging
from abc import abstractmethod
from random import shuffle
from typing import List, Dict

from ai_game.common import Board, Color, Action, State
from ai_game.player import Player

logger = logging.getLogger(__name__)


class Game(object):
    PLAYER_NUM = 2

    def __init__(self, player_list):
        self.player_list: List[Player] = player_list
        if len(player_list) != self.PLAYER_NUM:
            raise ValueError(f"player number:{len(player_list)} not same with expected:{self.PLAYER_NUM}")
        self.color2player: Dict[Color, Player] = dict()
        self.board = None
        self.cur_player = None

    @property
    def cur_color(self):
        return self.cur_player.color if self.cur_player else None

    @property
    def cur_state(self):
        return State(board=self.board, color=self.cur_color)

    @classmethod
    @abstractmethod
    def get_init_board(cls) -> Board:
        pass

    @classmethod
    @abstractmethod
    def get_init_state(cls) -> State:
        pass

    @classmethod
    @abstractmethod
    def get_valid_action_list(cls, state: State) -> List[Action]:
        pass

    @classmethod
    @abstractmethod
    def state_trans_func(cls, state: State, action: Action) -> State:
        pass

    @classmethod
    @abstractmethod
    def get_win_color(cls, state: State) -> Color:
        pass

    @classmethod
    @abstractmethod
    def is_terminate(cls, state: State) -> bool:
        pass

    @abstractmethod
    def start(self) -> Player:
        pass

    @abstractmethod
    def apply_action(self, action: Action):
        pass

    def get_winner(self) -> Player:
        win_color = self.get_win_color(self.cur_state)
        winner = self.color2player.get(win_color)
        return winner
