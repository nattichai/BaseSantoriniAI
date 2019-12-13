import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from .features import CNNFeature

class SantoriniNet():
    def __init__(self, board_dim: tuple, n_action: int, args):
        self.args = args
        self.net = Backbone(CNNFeature.n_ch, board_dim, n_action, args)
        self.action_size = n_action

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = [CNNFeature.extract(b) for b in input_boards]
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.net.model.fit(x=input_boards,
                           y=[target_pis, target_vs],
                           batch_size=self.args.batch_size,
                           epochs=self.args.epochs)

    def predict(self, board):
        # preparing input
        # feature extract
        board = CNNFeature.extract(board)
        board = board[np.newaxis, :, :, :].astype(np.float32)
        board = np.rollaxis(board, -1, 1)

        # predict
        pi, v = self.net.model.predict(board)

        return pi[0], v[0]
    
    def save(self, folder, fname):
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.net.model.save_weights(os.path.join(folder, fname))

    def load(self, folder, fname):
        self.net.model.load_weights(os.path.join(folder, fname))


class Backbone:
    def __init__(self, in_ch: int, board_dim: tuple, n_action: int, args):
        # game params
        self.board_x, self.board_y = board_dim
        self.action_size = n_action
        self.args = args

        n_ch = args.num_channels

        self.input = tf.keras.Input(shape=(self.board_x, self.board_y, in_ch))

        self.core = tf.keras.Sequential([
            layers.Conv2D(n_ch, 3, 1, 'same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n_ch, 3, 1, 'same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n_ch, 3, 1, 'same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(n_ch, 3, 1, 'same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(args.dropout),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(args.dropout),
        ])

        self.action_head = layers.Dense(n_action, activation='softmax')
        self.value_head = layers.Dense(1, activation='tanh')

        core = self.core(self.input)
        pi = self.action_head(core)
        v = self.value_head(core)

        self.model = tf.keras.Model(inputs=self.input, outputs=[pi, v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer=tf.keras.optimizers.Adam(args.lr))
