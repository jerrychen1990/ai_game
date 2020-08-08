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
import math

import tensorflow as tf
import logging
from tensorflow.keras.layers import Input, Convolution2D, Dense, Embedding, Flatten, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import categorical_crossentropy, mse

from ai_game.common import State, Action, Color, PutPieceAction
from typing import List, Tuple
import numpy as np

from ai_game.record import Record, ActionRecord

logger = logging.getLogger(__name__)


class StateModel:
    def __init__(self, color_list: List[Color]):
        self.color_list = color_list
        self.color_dict = {color: idx for idx, color in enumerate(color_list)}
        self.color_num = len(self.color_dict)
        self.nn_model: Model = None
        self.row_num = None
        self.col_num = None
        self.action_out_dim = None

    def build_nn_model(self, row_num, col_num, embedding_dim, filter_list=[]):
        logger.info("building nn model")
        board_input = Input(name="board_input", shape=(row_num, col_num))
        color_input = Input(name="color_input", shape=(1,))
        self.row_num = row_num
        self.col_num = col_num
        self.action_out_dim = row_num * col_num

        embedding_layer = Embedding(input_dim=self.color_num, output_dim=embedding_dim, name="color_embedding_layer",
                                    trainable=True, mask_zero=True, embeddings_initializer="uniform")

        board_embedding = embedding_layer(board_input)
        color_embedding = embedding_layer(color_input)
        color_embedding = Lambda(lambda x: tf.squeeze(x, axis=-2))(color_embedding)
        for filter_num, kernel_size in filter_list:
            board_embedding = Convolution2D(filters=filter_num, kernel_size=kernel_size)(board_embedding)

        board_embedding = Flatten()(board_embedding)
        total_embedding = Concatenate()([board_embedding, color_embedding])

        action_out = Dense(self.action_out_dim, activation="softmax", name="action_out")(total_embedding)
        value_out = Dense(1, activation="tanh", name="value_out")(total_embedding)

        self.nn_model = Model([board_input, color_input], [action_out, value_out])
        self.nn_model.summary()
        return self.nn_model

    def compile_nn_model(self):
        logger.info(f"compiling model")
        loss = {
            'action_out': categorical_crossentropy,
            'value_out': mse
        }
        loss_weights = {
            'action_out': 1.,
            'value_out': 1.
        }
        self.nn_model.compile(optimizer="adam", loss=loss, loss_weights=loss_weights, metrics=["accuracy"])

    def train(self, data: List[ActionRecord], epochs, batch_size, buffer_size=1024, **kwargs):
        logger.info(f"training with {len(data)} items")

        def generator_func():
            for record in data:
                item = self._record2tf_item(record)
                yield item

        dataset_type = dict(board_input=tf.int32, color_input=tf.int32), dict(action_out=tf.float32,
                                                                              value_out=tf.float32)
        dataset_shape = dict(board_input=(self.row_num, self.col_num,), color_input=()), dict(
            action_out=(self.action_out_dim,), value_out=())
        logger.info("loading dataset...")
        dataset = tf.data.Dataset.from_generator(generator_func, output_types=dataset_type,
                                                 output_shapes=dataset_shape)
        count = dataset.reduce(0, lambda x, _: x + 1).numpy()
        dataset = dataset.repeat().shuffle(buffer_size=buffer_size).batch(batch_size).prefetch(
            buffer_size=batch_size)
        logger.info(f"got dataset with {count} items...")

        if not kwargs.get("steps_per_epoch"):
            kwargs["steps_per_epoch"] = int(math.ceil(count / batch_size))

        logger.info("fitting...")
        self.nn_model.fit(dataset, epochs=epochs, **kwargs)
        return self.nn_model

    def _record2tf_item(self, record: ActionRecord, mode="train"):
        state = record.state
        input_dict = self._state2tf_item(state)
        action_prob_list = record.action_prob_list
        action_out = [0.] * self.action_out_dim
        for action, prob in action_prob_list:
            idx = action.row * self.row_num + action.col
            action_out[idx] = prob
        if record.win_color is None:
            value_out = 0
        else:
            value_out = 1. if state.color == record.win_color else -1.
        output_dict = dict(action_out=action_out, value_out=value_out)
        if mode == "train":
            return input_dict, output_dict
        return input_dict

    def _state2tf_item(self, state: State):
        board_input = []
        color_board = state.board.color_board
        for row in color_board:
            line = [self.color_dict.get(e) for e in row]
            board_input.append(line)
        color_input = self.color_dict[state.color]

        return dict(board_input=board_input, color_input=color_input)

    def _state2input(self, state: State):
        rs_dict = self._state2tf_item(state)
        return np.array([rs_dict['board_input']]), np.array([rs_dict['color_input']])

    def predict(self, state: State, valid_action_list: List[PutPieceAction]) -> Tuple[List[float], float]:
        board_input, color_input = self._state2input(state)
        action_out, value_out = self.nn_model.predict([board_input, color_input])
        action_prob = action_out[0]
        value = value_out[0][0]
        prob_list = [action_prob[action.row * self.col_num + action.col] for action in valid_action_list]

        return prob_list, value


def get_state_model_of_game(game_cls, build_kwargs):
    state_model = StateModel(color_list=[Color.BLANK] + game_cls.COLOR_LIST)
    state_model.build_nn_model(row_num=game_cls.ROW_NUM, col_num=game_cls.COL_NUM, **build_kwargs)
    state_model.compile_nn_model()
    return state_model
