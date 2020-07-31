import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from game_env import Game
from players import RandomPlayer, DeepQLearningAgent
import tensorflow as tf


# Train a DQN agent against self
print("Training a DQN against Self")
# some global variables
tf.random.set_seed(42)
board_size = 8
buffer_size=10000
gamma = 0.99
n_actions = 64
use_target_net = True
epsilon = 0.9
version = 'v1'
batch_size = 64
supervised = False
agent_type = 'DeepQLearningAgent'

# setup the game and players
p1 = DeepQLearningAgent(board_size=board_size, buffer_size=buffer_size,
                        gamma=gamma, n_actions=n_actions, 
                        use_target_net=use_target_net, epsilon=epsilon, 
                        version=version, name='dqn1')
p2 = DeepQLearningAgent(board_size=board_size, buffer_size=buffer_size,\
                        gamma=gamma, n_actions=n_actions, 
                        use_target_net=use_target_net, epsilon=epsilon,
                        version=version, name='dqn2')
p_random = RandomPlayer(board_size=board_size)
g = Game(player1=p1, player2=p2, board_size=board_size)
g2 = Game(player1=p1, player2=p_random, board_size=board_size)

# check the model architecture
print("Model architecture")
p1._model.summary()

# initializing parameters for DQN
reward_type = 'current'
sample_actions = False
decay = 0.85
epsilon_end = 0.1
n_games_buffer = 300
n_games_train = 10
episodes = 1 * (10**5)
log_frequency = 500
player_update_freq = 500

# Play some games initially to the agent's buffer
start_time = time.time()
for i in tqdm(range(n_games_buffer)):
    g.reset()
    _ = g.play(add_to_buffer=True)
print("Playing %{:4d} games took %{:4d}s".format(n_games_buffer, 
                                     int(time.time() - start_time)))

model_logs = {'iteration':[], 'reward_mean':[], 'loss':[]}

# train the agent
print("Starting training against self")
for idx in tqdm(range(episodes)):
    # play game and add to buffer
    _ = g.play(add_to_buffer=True)

    # train
    loss = p1.train_agent(batch_size=batch_size)
    _ = p2.train_agent(batch_size=batch_size)

    # select the better player
    if(idx % player_update_freq == 0):
        win_1 = 0
        # play 10 games and check how many times p1 wins
        for j in range(20):
            winner = g.play()
            coins = g.get_players_coin()
            if(winner != -1 and coins[winner].name == "dqn1"):
                win_1 += 1;
        # select the better one from p1 and p2
        if(win_1 == 5):
            pass
        elif(win_1 > 5):
            p2.copy_weights_from_agent(p1)
        else:
            p1.copy_weights_from_agent(p2)


    # epsilon decay and target_net
    # saving buffers and models 
    if idx % log_frequency == 0:
        model_logs['iteration'].append(idx+1)
        # play games agains random player for evaluation
        win_1 = 0
        for j in range(20):
            winner = g2.play()
            coins = g2.get_players_coin()
            if(winner != -1 and coins[winner].name == "dqn1"):
                win_1 += 1;
        model_logs['reward_mean'].append(round(win_1/20.0, 2))
        model_logs['loss'].append(loss)
        pd.DataFrame(model_logs)[['iteration', 'reward_mean','loss']]\
          .to_csv('model_logs/{:s}.csv'.format(version), index=False)
        
        # update target networks
        p1.update_target_net()
        p2.update_target_net()
        
        # save the models
        p1.save_model(file_path='models/{:s}'.format(version), 
                      iteration=(int(idx / (n_games_train))))
        # keep some epsilon alive for training
        p1.epsilon = max(p1.epsilon * decay, epsilon_end)
