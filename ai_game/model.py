# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model
   Description :
   Author :       chenhao
   date：          2020/8/1
-------------------------------------------------
   Change Activity:
                   2020/8/1:
-------------------------------------------------
"""

import tensorflow as tf
from tensorflow.keras.layers import Input, Convolution2D, Dense, Embedding, Flatten, Concatenate, Lambda
from tensorflow.keras.models import Model

from ai_game.common import State, Action, Color, PutPieceAction
from typing import List
import numpy as np


class StateModel:
    def __init__(self, color_list: List[Color]):
        self.color_list = color_list
        self.color_dict = {color: (idx + 1) for idx, color in enumerate(color_list)}
        self.nn_model: Model = None
        self.row_num = None
        self.col_num = None

    def build_nn_model(self, row_num, col_num, color_num, embedding_dim, filter_list=[]):
        board_input = Input(name="board_input", shape=(row_num, col_num))
        color_input = Input(name="color_input", shape=(1,))
        self.row_num = row_num
        self.col_num = col_num
        action_out_dim = row_num * col_num

        embedding_layer = Embedding(input_dim=color_num, output_dim=embedding_dim, name="color_embedding_layer",
                                    trainable=True, mask_zero=True, embeddings_initializer="uniform")

        board_embedding = embedding_layer(board_input)
        color_embedding = embedding_layer(color_input)
        color_embedding = Lambda(lambda x: tf.squeeze(x, axis=-2))(color_embedding)
        for filter_num, kernel_size in filter_list:
            board_embedding = Convolution2D(filters=filter_num, kernel_size=kernel_size)(board_embedding)

        board_embedding = Flatten()(board_embedding)
        total_embedding = Concatenate()([board_embedding, color_embedding])

        action_out = Dense(action_out_dim, activation="softmax", name="action_out")(total_embedding)
        value_out = Dense(1, activation="sigmoid", name="value_out")(total_embedding)

        self.nn_model = Model([board_input, color_input], [action_out, value_out])
        self.nn_model.summary()

        return self.nn_model

    def _state2input(self, state: State):
        board_input = []
        board = state.board.board
        for row in board:
            line = [self.color_dict.get(e.color) if e else 0 for e in row]
            board_input.append(line)
        board_input = np.array([board_input])

        color_input = np.array([self.color_dict[state.color]])
        return board_input, color_input

    def predict(self, state: State, valid_action_list: List[PutPieceAction]):
        board_input, color_input = self._state2input(state)
        action_out, value_out = self.nn_model.predict([board_input, color_input])
        action_prob = action_out[0]
        value = value_out[0][0]

        action_prob_dict = {action: action_prob[action.row * self.col_num + action.col] for action in valid_action_list}

        return action_prob_dict, value
