import numpy as np

class MCTS:
    def __init__(self, game, net, n_sims, c_puct):
        self.game = game
        self.net = net
        self.n_sims = n_sims
        self.c_puct = c_puct

        self.Qsa = {}
        self.Nsa = {}
        self.Qs = {}
        self.Ns = {}
        self.Ps = {}
        self.Vs = {}

    def get_action_prob(self, state, temp=1):
        board = state[0]
        workers = np.concatenate([[np.where(state[1] == i)] for i in [-1, -2, 1, 2]])[:, :, 0]
        parts = state[2].diagonal()
        current_player = -1
        done = False
        
        for sim in range(self.n_sims):
            self.game.set_state(board, workers, parts, current_player, done)
            self.search(state)
        
        s = self.game.tostring(state)
        
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.action_size)]
        
        if temp == 0:
            best_action = np.argmax(counts)
            probs = np.zeros(self.game.action_size)
            probs[best_action] = 1
        else:
            counts = np.array([c ** (1. / temp) for c in counts])
            probs = counts / float(sum(counts))

        return probs

    def search(self, state):
        s = self.game.tostring(state)
        
        if s not in self.Ns:
            self.Qs[s] = 0
            self.Ns[s] = 0
            self.Vs[s] = self.game.legal_moves()
            
            pi, v = self.net.predict(state)
            mask = np.zeros(self.game.action_size)
            mask[self.Vs[s]] = 1
            pi *= mask
            if sum(pi) > 0:
                pi /= sum(pi)
            else:
                print('Warning: all predicted moves are illegal')
                pi = mask
                pi /= sum(pi)
            self.Ps[s] = pi
            return -v
        
        valid_moves = self.Vs[s]
        best_action = -1
        best_qu = -np.inf
        
        for a in valid_moves:
            if (s, a) in self.Qsa:
                q = self.Qsa[(s, a)]
                u = self.c_puct * self.Ps[s][a] * (self.Ns[s] ** 0.5) / (1 + self.Nsa[(s, a)])
            else:
                q = 0
                u = self.c_puct * self.Ps[s][a] * ((self.Ns[s] + 1e-8) ** 0.5)
            if q + u > best_qu:
                best_qu = q + u
                best_action = a
            
        a = best_action
        state, reward, done, _ = self.game.step(a)
        if done:
            v = reward
        else:
            v = self.search(state)
        
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Qsa[(s, a)] * self.Nsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
            
        self.Qs[s] = (self.Qs[s] * self.Ns[s] + v) / (self.Ns[s] + 1)
        self.Ns[s] += 1

        return -v