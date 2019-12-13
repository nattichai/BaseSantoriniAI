import os
import numpy as np
from tqdm import tqdm_notebook
from .mcts import MCTS

class Coach:
    def __init__(self, make_game, net, args):
        self.make_game = make_game
        self.net = net
        self.args = args
        
        self.game = make_game()
        self.history_train_data = []
    
    def learn(self):
        for itr in range(1, self.args.n_itr + 1):
            print('------ITER ' + str(itr) + '------')

            self.history_train_data.append([])
            
            for eps in tqdm_notebook(range(self.args.n_ep)):
                self.run_episode()
            
            while len(self.history_train_data) > self.args.max_history_itr:
                print(f'train data size exceeds {self.args.max_history_itr} iters -> remove oldest data')
                self.history_train_data.pop(0)
                
            train_data = []
            for data in self.history_train_data:
                train_data.extend(data)
                
            self.net.train(train_data)
            self.net.save(self.args.checkpoint, f'model{itr}.h5')
                    
    def run_episode(self):
        train_data = []
        mcts = MCTS(self.make_game(), self.net, self.args.n_sims, self.args.c_puct)
        
        step = 0
        state = self.game.reset()
        
        while True:
            step += 1
            pi = mcts.get_action_prob(state, temp=int(step < self.args.no_temp_step))
            
            train_data.append((state, pi, self.game.current_player))
            
            action = np.random.choice(len(pi), p=pi)
            state, reward, done, _ = self.game.step(action)
            
            if done:
                for state, pi, player in train_data:
                    if player == self.game.current_player:
                        v = reward
                    else:
                        v = -reward
                    self.history_train_data[-1].append((state, pi, v))
                return