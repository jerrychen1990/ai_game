# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     player
   Description :
   Author :       chenhao
   date：          2020/7/29
-------------------------------------------------
   Change Activity:
                   2020/7/29:
-------------------------------------------------
"""
import logging
from typing import List
from random import choice
from abc import abstractmethod
from ai_game.common import Action, Color, PutPieceAction, Piece

logger = logging.getLogger(__name__)


class Player(object):
    def __init__(self, name):
        self.name = name
        self.color = None

    @abstractmethod
    def choose_action(self, action_list: List[Action]) -> Action:
        assert len(action_list) > 0

    def set_color(self, color: Color):
        self.color = color

    def __str__(self):
        return self.name


class RandomPlayer(Player):
    def __init__(self, name):
        self.name = name

    def choose_action(self, action_list: List[Action]) -> Action:
        return choice(action_list)


class HumanPlayer(Player):
    def __init__(self, name):
        self.name = name

    def choose_action(self, action_list: List[Action]) -> Action:
        while True:
            try:
                ipt = input("input row and col, split with space:")
                row, col = ipt.split(" ")
                row = int(row)
                col = int(col)
                action = PutPieceAction(row=row, col=col, piece=Piece(color=self.color))
                if action in action_list:
                    return action
                else:
                    raise Exception(f"action:{action} not in valid action list!")

            except Exception as e:
                logger.warning(e)
                logger.info(f"invalid input {ipt}")
