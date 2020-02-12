from game_env import StateEnv, Game, StateEnvBitBoard, StateConverter
from players import RandomPlayer
import numpy as np
import time
from tqdm import tqdm

board_size = 8

# initialize classes
p1 = RandomPlayer(board_size=board_size)
p2 = RandomPlayer(board_size=board_size)
g = Game(player1=p1, player2=p2, board_size=board_size)

def convert_boards(s, m=None):
    """Convert the board state from bitboard to ndarray

    Parameters
    ----------
    s : list
        contains black, white bitboards and current player
    m : int (64 bit), default None
        bitboard for legal moves
    
    Returns
    -------
    s : list
        contains black, white board arrays and current player
    m : ndarray
        legal moves array
    """
    s[0] = conv.convert(s[0], 
               input_format='bitboard_single',
               output_format='ndarray')
    s[1] = conv.convert(s[1], 
               input_format='bitboard_single',
               output_format='ndarray')
    if(m is not None):
        m = conv.convert(m, 
               input_format='bitboard_single',
               output_format='ndarray')
        return s, m
    # else return just s
    return s

def compare_boards(s, correct_s, case_no):
    """Compare boards from the game environment and the correct versions

    Parameters
    ----------
    s : list
        [board state, legal moves, current player]
        board state = [black board array, white board array, current player]
        when step function is applied, done should also be checked
    correct_s : list
        same format as above but all values are correct
    case_no : int
        the current case being evaluated, used for printing
    """
    # check match of black board
    if((s[0][0] == correct_s[0][0]).sum() != 64):
        print('Case {:d}: Black board not matching'.format(case_no))
        print('Expected')
        print(correct_s[0][0])
        print('Got')
        print(s[0][0])
    # check match of white board
    if((s[0][1] == correct_s[0][1]).sum() != 64):
        print('Case {:d}: White board not matching'.format(case_no))
        print('Expected')
        print(correct_s[0][1])
        print('Got')
        print(s[0][1])
    # check match of current player
    if(s[0][2] != correct_s[0][2]):
        print('Case {:d}: Starting player in board state is not {:d} but {:d}'\
              .format(case_no, correct_s[0][2], s[0][2]))
    # check match of legal moves
    if((s[1] == correct_s[1]).sum() != 64):
        print('Case {:d}: Legal moves not matching'.format(case_no))
        print('Expected')
        print(correct_s[1])
        print('Got')
        print(s[1])
    # check match of current player
    if(s[2] != correct_s[2]):
        print('Case {:d}: Starting player in board state is not {:d} but {:d} (last argument)'\
              .format(case_no, correct_s[2], s[2]))
    # check match of done
    if(len(s) == 4):
        if(s[3] != correct_s[3]):
            print('Case {:d}: Done should be {:d} but got {:d}'\
                  .format(case_no, correct_s[3], s[3]))

# test cases to check if the environment is working correctly
while(1):
    env = StateEnvBitBoard(board_size=board_size)
    conv = StateConverter()
    ######## Case 1 : reset is correct ########
    print('Running Case 1: Env reset')
    s, legal_moves, player = env.reset()
    # convert s to required format
    s, legal_moves = convert_boards(s, legal_moves)
    # correct outputs
    correct_black = np.zeros((8,8), dtype=np.uint8)
    correct_black[[3,4], [4,3]] = 1
    correct_white = np.zeros((8,8), dtype=np.uint8)
    correct_white[[3,4], [3,4]] = 1
    correct_legal_moves = np.zeros((8,8), dtype=np.uint8)
    correct_legal_moves[[2,3,4,5], [4,5,2,3]] = 1
    correct_player = 1
    # do the comparisons stepwise
    compare_boards([s, legal_moves, player], 
                   [[correct_black, correct_white, correct_player], \
                            correct_legal_moves, correct_player], 
                    1)

    ######## Case 2 : playing a move on initial board ########
    print('Running Case 2: Playing a move on initial board')
    s, _, _ = env.reset()
    next_s, next_legal_moves, next_player, done = env.step(s, 1<<34)
    next_s, next_legal_moves = convert_boards(next_s, next_legal_moves)
    correct_black = np.zeros((8,8), dtype=np.uint8)
    correct_black[4,3] = 1
    correct_white = np.zeros((8,8), dtype=np.uint8)
    correct_white[[3,3,3,4], [3,4,5,4]] = 1
    correct_legal_moves = np.zeros((8,8), dtype=np.uint8)
    correct_legal_moves[[2,2,4], [3,5,5]] = 1
    correct_player = 0
    # do the comparisons stepwise
    compare_boards([next_s, next_legal_moves, next_player, done], 
                   [[correct_black, correct_white, correct_player], \
                        correct_legal_moves, correct_player, 0], 
                    2)


    # break from the while loop
    break


# check time taken to play 1000 games
"""
total_games = 1000
time_list = np.zeros(total_games)
for i in tqdm(range(total_games)):
    start_time = time.time()
    g.reset()
    winner = g.play()
    time_list[i] = time.time() - start_time
# print results
print('Total time taken to play {:d} games : {:.3f}s'.format(total_games, time_list.sum()))
print('Average time per game : {:.3f}s +- {:.3f}s'.format(np.mean(time_list), np.std(time_list)))
"""

# record and save game
# for i in range(6, 11):
    # g.record_gameplay('images/gameplay_random_{:d}.mp4'.format(i))
