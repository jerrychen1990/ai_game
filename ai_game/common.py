# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     common
   Description :
   Author :       chenhao
   date：          2020/7/29
-------------------------------------------------
   Change Activity:
                   2020/7/29:
-------------------------------------------------
"""
from enum import Enum
from random import choice

from pydantic import BaseModel
from pydantic.dataclasses import dataclass

from ai_game.constant import *
from typing import List, Tuple


class Color(str, Enum):
    BLANK = "_"
    BLACK = "B"
    WHITE = "W"
    X = "X"
    O = "O"


class Piece(BaseModel):
    color: Color

    def __repr__(self):
        return self.color.value

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))


class Board(object):

    def __init__(self, row_num, col_num):
        self.row_num = row_num
        self.col_num = col_num
        self.board = [[None] * self.col_num for _ in range(self.row_num)]

    @property
    def rows(self):
        return self.board

    @property
    def cols(self):
        return [[self.board[row][col] for row in range(self.row_num)] for col in range(self.col_num)]

    @property
    def diagonals(self):
        diagonals = []
        if self.row_num == self.col_num:
            diagonals.append([self.board[i][i] for i in range(self.row_num)])
            diagonals.append([self.board[i][self.row_num - 1 - i] for i in range(self.row_num)])
        return diagonals

    def __repr__(self):
        row_list = []
        for row in self.rows:
            row_list.append("".join(str(e) if e else EMPTY for e in row))
        rs_str = "\n".join(row_list)
        return rs_str

    def __str__(self):
        return self.__repr__()

    def set_piece(self, row, col, piece: Piece):
        self.board[row][col] = piece


def get_empty_positions(board: Board) -> List[Tuple[int, int]]:
    rs_list = []
    for row_idx, row in enumerate(board.rows):
        for col_idx, ele in enumerate(row):
            if not ele:
                rs_list.append((row_idx, col_idx))
    return rs_list


def is_full(board: Board) -> bool:
    empty_positions = get_empty_positions(board)
    return len(empty_positions) == 0


class State(BaseModel):
    board: Board
    color: Color

    def __repr__(self):
        rs_str = self.color.value + "\n" + self.board.__str__()
        return rs_str

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash((str(self.board), self.color))

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return hash(other) == self.__hash__()


class Action(BaseModel):
    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))


class PutPieceAction(Action):
    row: int
    col: int
    piece: Piece


def put_piece(board: Board, action: PutPieceAction):
    board.set_piece(action.row, action.col, action.piece)


class Strategy:
    @staticmethod
    def random_choose(state: State, action_list: List[Action]) -> Action:
        return choice(action_list)
