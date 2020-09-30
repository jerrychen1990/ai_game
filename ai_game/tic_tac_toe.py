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

import copy
import logging
from random import shuffle
from typing import List

from ai_game.common import Board, Color, Action, get_empty_positions, PutPieceAction, Piece, State, put_piece, is_full
from ai_game.game import Game
from ai_game.player import Player
from ai_game.util import get_logger


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
    ROW_NUM = 3
    COL_NUM = 3
    PLAYER_NUM = 2
    COLOR_LIST = [Color.X, Color.O]

    def __init__(self, player_list, recorder=None, level=logging.INFO):
        super().__init__(player_list)
        self.board = None
        self.recorder = recorder
        self.logger = get_logger(self.__class__.__name__, level)

    def _init_board(self):
        self.logger.debug("initializing board...")
        self.board = self.get_init_board()

    def _assign_color(self):
        self.logger.debug("assigning color...")
        shuffle(self.player_list)
        assert len(self.COLOR_LIST) == len(self.player_list)
        for player, color in zip(self.player_list, self.COLOR_LIST):
            self.logger.debug(f"player:{player.name} get color:{color}")
            player.set_color(color)
            self.color2player[color] = player

    @classmethod
    def get_init_board(cls) -> Board:
        board = Board.init_board(cls.ROW_NUM, cls.COL_NUM)
        return board

    @classmethod
    def get_init_state(cls) -> State:
        board = cls.get_init_board()
        color = cls.COLOR_LIST[0]
        state = State(board=board, color=color)
        return state

    @classmethod
    def get_win_color(cls, state: State) -> Color:
        board = state.board
        all_lines = board.rows + board.cols + board.diagonals
        for line in all_lines:
            color = get_line_same_color(line)
            if color:
                return color
        return None

    @classmethod
    def get_valid_action_list(cls, state: State) -> List[Action]:
        board = state.board
        empty_positions = get_empty_positions(board)
        action_list = [PutPieceAction(row=r, col=c, piece=Piece(color=state.color)) for r, c in empty_positions]
        return action_list

    @classmethod
    def is_terminate(cls, state: State) -> bool:
        if cls.get_win_color(state):
            return True
        return is_full(state.board)

    @classmethod
    def _get_next_color(cls, color: Color) -> Color:
        color_idx = cls.COLOR_LIST.index(color)
        next_idx = (color_idx + 1) % len(cls.COLOR_LIST)
        return cls.COLOR_LIST[next_idx]

    @classmethod
    def state_trans_func(cls, state: State, action: PutPieceAction) -> State:
        next_board = copy.deepcopy(state.board)
        put_piece(next_board, action)
        next_color = cls._get_next_color(state.color)
        next_state = State(board=next_board, color=next_color)
        return next_state

    def apply_action(self, action: PutPieceAction):
        put_piece(self.board, action)

    def process_winner(self, winner: Player):
        if winner:
            self.logger.info(f"game over, winner is :{winner.name}")
            win_color = winner.color
        else:
            self.logger.info("game over, this is a draw game!")
            win_color = None
        if self.recorder:
            self.recorder.update_win_color(win_color)

    def start(self):
        self.logger.info("game starts")
        self._init_board()
        self.logger.debug("current board:\n" + str(self.board))
        self._assign_color()
        round = 0
        winner = None

        while not winner:
            skip_num = 0
            self.logger.debug(f"round:{round + 1}")
            for player in self.player_list:
                self.cur_player = player
                winner = self.get_winner()
                if winner:
                    break
                self.logger.debug(f"player:{player.name}'s action")
                action_list = self.get_valid_action_list(self.cur_state)
                if not action_list:
                    self.logger.debug(f"player:{player.name} has no valid action, skip")
                    skip_num += 1
                else:
                    action = player.choose_action(self.cur_state, action_list, round=round)
                    self.logger.info(f"player:{player.name} choose action:{action}")
                    self.apply_action(action)
                self.logger.debug("current board:\n" + str(self.board))

            if skip_num == len(self.player_list):
                self.logger.debug(f"all player skip, game ends")
                break
            round += 1
        self.process_winner(winner)
        return winner
