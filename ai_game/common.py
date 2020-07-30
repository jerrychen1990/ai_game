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

from pydantic import BaseModel
from ai_game.constant import *
from typing import List, Tuple


class Color(Enum):
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


class Board(object):
    def __init__(self, row_num, col_num=None):
        self.row_num = row_num
        self.col_num = col_num if col_num else row_num
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


class Action(BaseModel):
    pass


class PutPieceAction(Action):
    row: int
    col: int
    piece: Piece
