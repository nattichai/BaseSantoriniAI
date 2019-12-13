import numpy as np

class CNNFeature:
    """feature engineering for santorini"""
    n_ch = 5

    @classmethod
    def get_buildings_layer(cls, board):
        buildings_layer = board[0, :, :].copy()
        return (buildings_layer)

    @classmethod
    def get_worker_layer(cls, board, worker):
        idx = np.where(board[1, :, :] == worker)
        worker_layer = np.zeros_like(board[0])
        worker_layer[idx] = 1
        return (worker_layer)

    @classmethod
    def extract(cls, board):
        return np.stack([
            cls.get_buildings_layer(board),
            cls.get_worker_layer(board, -1),
            cls.get_worker_layer(board, -2),
            cls.get_worker_layer(board, 1),
            cls.get_worker_layer(board, 2),
        ])
