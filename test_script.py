from game_env import StateEnv, Game
from players import RandomPlayer
import numpy as np

board_size = 8

# initialize classes
p1 = RandomPlayer(board_size=board_size)
p2 = RandomPlayer(board_size=board_size)
g = Game(player1=p1, player2=p2, board_size=board_size)

# g.reset()
# g.play()

# record and save game
for i in range(5):
    g.record_gameplay('images/gameplay_random_{:d}.mp4'.format(i))
