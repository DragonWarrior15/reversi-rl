from game_env import StateEnv, Game
from players import RandomPlayer
import numpy as np
import time
from tqdm import tqdm

board_size = 8

# initialize classes
p1 = RandomPlayer(board_size=board_size)
p2 = RandomPlayer(board_size=board_size)
g = Game(player1=p1, player2=p2, board_size=board_size)

# check time taken to play 10000 games
"""
total_games = 1000
time_list = np.zeros(total_games)
for i in tqdm(range(total_games)):
    start_time = time.time()
    g.reset()
    winner = g.play()
    time_list[i] = time.time() - start_time
# print results
print('Total time take to play {:d} games : {:.3f}s'.format(total_games, time_list.sum()))
print('Average time per game : {:.3f}s +- {:.3f}s'.format(np.mean(time_list), np.std(time_list)))
"""

# record and save game
for i in range(6, 11):
    g.record_gameplay('images/gameplay_random_{:d}.mp4'.format(i))
