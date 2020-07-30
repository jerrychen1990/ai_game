# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     tic_tac_toe
   Description :
   Author :       chenhao
   date：          2020/7/29
-------------------------------------------------
   Change Activity:
                   2020/7/29:
-------------------------------------------------
"""

import logging
from random import shuffle
from typing import List

from ai_game.common import Board, Color, Action, get_empty_positions, PutPieceAction, Piece
from ai_game.game import Game

logger = logging.getLogger(__name__)


def get_line_same_color(line: List[Piece]) -> Color:
    pre_color = None
    for piece in line:
        if not piece:
            return None
        if pre_color and pre_color != piece.color:
            return None
        pre_color = piece.color
    return pre_color


class TicTacToe(Game):
    BOARD_SIZE = 3
    PLAYER_NUM = 2
    COLOR_LIST = [Color.X, Color.O]

    def __init__(self, player_list):
        super().__init__(player_list)

        self.board = None

    def _init_board(self):
        logger.info("initializing board...")
        self.board = Board(self.BOARD_SIZE)

    def _assign_color(self):
        logger.info("assigning color...")
        shuffle(self.player_list)
        assert len(self.COLOR_LIST) == len(self.player_list)
        for player, color in zip(self.player_list, self.COLOR_LIST):
            logger.info(f"player:{player.name} get color:{color}")
            player.set_color(color)
            self.color2player[color] = player

    def get_valid_actions(self, color: Color) -> List[Action]:
        empty_positions = get_empty_positions(self.board)
        action_list = [PutPieceAction(row=r, col=c, piece=Piece(color=color)) for r, c in empty_positions]
        return action_list

    def get_win_color(self):
        all_lines = self.board.rows + self.board.cols + self.board.diagonals
        for line in all_lines:
            color = get_line_same_color(line)
            if color:
                return color
        return None

    def apply_action(self, action: PutPieceAction):
        self.board.set_piece(action.row, action.col, action.piece)

    def start(self):
        logger.info("game starts")
        self._init_board()
        logger.info("current board:\n" + str(self.board))
        self._assign_color()
        round = 0
        winner = None

        while not winner:
            skip_num = 0
            logger.info(f"round:{round + 1}")
            for player in self.player_list:
                winner = self.get_winner()
                if winner:
                    break
                logger.info(f"player:{player.name}'s action")
                action_list = self.get_valid_actions(player.color)
                if not action_list:
                    logger.info(f"player:{player.name} has no valid action, skip")
                    skip_num += 1
                else:
                    action = player.choose_action(action_list)
                    logger.info(f"player:{player.name} choose action:{action}")
                    self.apply_action(action)
                logger.info("current board:\n" + str(self.board))

            if skip_num == len(self.player_list):
                logger.info(f"all player skip, game ends")
                break
            round += 1
        if winner:
            logger.info(f"game over, winner is :{winner.name}")
        else:
            logger.info("game over, this is a draw game!")
        return winner
