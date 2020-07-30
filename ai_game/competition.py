# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     competition
   Description :
   Author :       chenhao
   date：          2020/7/30
-------------------------------------------------
   Change Activity:
                   2020/7/30:
-------------------------------------------------
"""
import logging
from ai_game.constant import *

from ai_game.game import Game

logger = logging.getLogger(__name__)


class Competition:
    def __init__(self, game_cls, player_list, match_num):
        self.game_cls = game_cls
        self.player_list = player_list
        self.match_num = match_num
        self.score_board = {player: 0 for player in self.player_list}
        self.score_board[DRAW] = 0

    def show_score_board(self):
        logger.info("current scoreboard:")

        for k, v in sorted(self.score_board.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"{str(k):25}:\t{v}")

    def start(self):
        for idx in range(self.match_num):
            game: Game = self.game_cls(self.player_list)
            winner = game.start()
            if not winner:
                winner = DRAW
            self.score_board[winner] += 1
        logger.info(f"{self.match_num} matches ends")
        self.show_score_board()
