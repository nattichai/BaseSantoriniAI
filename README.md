# BaseSantoriniAI
A baseline AlphaZero AI for Santorini board game

## Usage
### Training
See `train.ipynb`
```Python
...

game = make_game()
net = SantoriniNet(game.board_dim, game.action_size, NetArgs)
coach = Coach(make_game, net, CoachArgs)
coach.learn()
```

### Model usage
```Python
...
from alphazero.players import *
...

net.load('checkpoint/', 'model1.h5')
alpha_player = AlphaPlayer(make_game, net, n_sims=10, c_puct=1)

state = env.reset()
action = alpha_player.play(state)
```

### Online arena
See both `arena1.ipynb` and `arena2.ipynb`. and also [chula_rl online](https://github.com/phizaz/chula_rl/tree/master/chula_rl/online) for more details

_Note: need to pull latest chula_rl first_

## What can be improved
### Neural network
Currently, it is a CNN from [chula_rl solution](https://github.com/phizaz/chula_rl/tree/master/chula_rl/alphazero/santorini/net). Edit files in `net/` folder
#### Some ideas
- **make the model predict faster** - it can increase training speed by **~10 times**. see **_TensorRT_** or **_Tensorflow Lite_**
- change nn architecture
- L2 loss
- validation splitting
- callbacks ex. model checkpoint, early stop, learning rate scheduler, cyclical learning rate
- position averaging - deduplicate repeated training data by averaging pi and v

### Hyperparameter tuning
Edit `CoachArgs` and `NetArgs` in `train.ipynb`
- n_ep: number of self-play episodes(games) per itr
- n_sims: number of simulations per move in MCTS
- c_puct: exploration constant
- no_temp_step: turn that it will choose action deterministically (before is randomly choosing from pi)
- max_history_itr: maximum number of training data to keep

### AlphaZero algorithm
See a lot of ideas in [this medium serie](https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191) ([overview](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a)). Edit file `alphazero/coach` and `alphazero/mcts`.
#### Some ideas
- add dirichlet noise - [explaination](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5), [implementation](https://github.com/evg-tyurin/alpha-nagibator/blob/48b2ebd3ca272f388c13277297edbb60d98eb64b/MCTS.py#L191)
- change training target - [explaination](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628)
- slow window - [explaination](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a)
- augment training data by symmetrical positions
