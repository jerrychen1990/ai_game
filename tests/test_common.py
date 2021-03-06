# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_common
   Description :
   Author :       chenhao
   date：          2020/7/29
-------------------------------------------------
   Change Activity:
                   2020/7/29:
-------------------------------------------------
"""

import unittest
from ai_game.common import *


class TestCommon(unittest.TestCase):
    def test_board(self):
        board = Board(3)
        print(board)
        board.set_piece(0, 0, Piece(color=Color.BLACK))
        board.set_piece(1, 1, Piece(color=Color.BLACK))
        board.set_piece(1, 2, Piece(color=Color.WHITE))

        print("rows")
        print(board.rows)
        print("cols")
        print(board.cols)
        print("diagonals")
        print(board.diagonals)

        empty_positions = get_empty_positions(board)
        print(empty_positions)

        self.assertEqual(6, len(empty_positions))
        print(board)

