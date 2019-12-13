import numpy as np
from .mcts import MCTS

class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, state):
        return np.random.choice(self.game.legal_moves())

class HumanPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, state):
        while True:
            a = input()
            worker, move, build = a.split()
            a = self.game.atoi[(int(worker), move, build)]
            if a in self.game.legal_moves():
                break
            else:
                print('Invalid move')
        return a

class AlphaPlayer:
    def __init__(self, make_game, net, n_sims=100, c_puct=1):
        self.game = make_game()
        self.net = net
        self.mcts = MCTS(make_game(), net, n_sims=n_sims, c_puct=c_puct)

    def play(self, state):
        return np.argmax(self.mcts.get_action_prob(state, temp=0))