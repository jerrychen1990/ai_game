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

from ai_game.common import Board, Color, Action
from ai_game.player import Player

logger = logging.getLogger(__name__)


class Game(object):
    BOARD_SIZE = 3
    PLAYER_NUM = 2

    def __init__(self, player_list):
        self.player_list: List[Player] = player_list
        if len(player_list) != self.PLAYER_NUM:
            raise ValueError(f"player number:{len(player_list)} not same with expected:{self.PLAYER_NUM}")
        self.color2player: Dict[Color, Player] = dict()

    @abstractmethod
    def start(self) -> Player:
        pass

    @abstractmethod
    def get_win_color(self) -> Color:
        pass

    def get_winner(self) -> Player:
        win_color = self.get_win_color()
        winner = self.color2player.get(win_color)
        return winner

    @abstractmethod
    def get_valid_actions(self, color: Color) -> List[Action]:
        pass

    @abstractmethod
    def apply_action(self, action: Action):
        pass
