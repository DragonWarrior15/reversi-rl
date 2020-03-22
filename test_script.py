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

    Returns
    -------
    success : bool
        whether all test cases passed
    """
    success = True
    # check match of black board
    if((s[0][0] == correct_s[0][0]).sum() != 64):
        success = False
        print('Case {:d}: Black board not matching'.format(case_no))
        print('Expected')
        print(correct_s[0][0])
        print('Got')
        print(s[0][0])
    # check match of white board
    if((s[0][1] == correct_s[0][1]).sum() != 64):
        success = False
        print('Case {:d}: White board not matching'.format(case_no))
        print('Expected')
        print(correct_s[0][1])
        print('Got')
        print(s[0][1])
    # check match of current player
    if(s[0][2] != correct_s[0][2]):
        success = False
        print('Case {:d}: Player in board state is not {:d} but {:d}'\
              .format(case_no, correct_s[0][2], s[0][2]))
    # check match of legal moves
    if((s[1] == correct_s[1]).sum() != 64):
        success = False
        print('Case {:d}: Legal moves not matching'.format(case_no))
        print('Expected')
        print(correct_s[1])
        print('Got')
        print(s[1])
    # check match of current player
    if(s[2] != correct_s[2]):
        success = False
        print('Case {:d}: Starting player in board state is not {:d} but {:d} (last argument)'\
              .format(case_no, correct_s[2], s[2]))
    # check match of done
    if(len(s) == 4):
        if(s[3] != correct_s[3]):
            success = False
            print('Case {:d}: Done should be {:d} but got {:d}'\
                  .format(case_no, correct_s[3], s[3]))
    # return
    return success

# test cases to check if the environment is working correctly
while(1):
    print('RUNNING TEST CASES FOR GAME ENVIRONMENT')
    success = True
    env = StateEnvBitBoard(board_size=board_size)
    conv = StateConverter()
    ######## Case 1: reset is correct ########
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
    success = success & compare_boards([s, legal_moves, player], 
                   [[correct_black, correct_white, correct_player], \
                            correct_legal_moves, correct_player], 
                    1)

    ######## Case 2: playing a move on initial board ########
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
    success = success & compare_boards([next_s, next_legal_moves, next_player, done], 
                   [[correct_black, correct_white, correct_player], \
                        correct_legal_moves, correct_player, 0], 
                    2)

    ######## Case 3: a move which results in all but one black ########
    print('Running Case 3: Playing a move on custom board')
    s = [((1<<61)-1)<<3, 1<<2, 0]
    next_s, next_legal_moves, next_player, done = env.step(s, 1<<1)
    next_s, next_legal_moves = convert_boards(next_s, next_legal_moves)
    correct_black = np.ones((8,8), dtype=np.uint8)
    correct_black[7,7] = 0
    correct_white = np.zeros((8,8), dtype=np.uint8)
    correct_legal_moves = np.zeros((8,8), dtype=np.uint8)
    correct_player = 0
    # do the comparisons stepwise
    success = success & compare_boards([next_s, next_legal_moves, next_player, done], 
                   [[correct_black, correct_white, correct_player], \
                        correct_legal_moves, correct_player, 1], 
                    3)

    ######## Case 4: a move which results in all but one white ########
    print('Running Case 4: Playing a move on custom board')
    s = [1<<2, ((1<<61)-1)<<3, 1]
    next_s, next_legal_moves, next_player, done = env.step(s, 1<<1)
    next_s, next_legal_moves = convert_boards(next_s, next_legal_moves)
    correct_white = np.ones((8,8), dtype=np.uint8)
    correct_white[7,7] = 0
    correct_black = np.zeros((8,8), dtype=np.uint8)
    correct_legal_moves = np.zeros((8,8), dtype=np.uint8)
    correct_player = 1
    # do the comparisons stepwise
    success = success & compare_boards([next_s, next_legal_moves, next_player, done], 
                   [[correct_black, correct_white, correct_player], \
                        correct_legal_moves, correct_player, 1], 
                    4)

    ######## Case 5: custom board from a match ########
    # https://www.worldothello.org/about/world-othello-championship/woc/2017/round/5
    # Maria Serena Vecchi 13-51 Caroline Nicolas, move 53 - 54
    print('Running Case 5: Custom board from a match')
    s = [9151597030240616704, 35183259641773180, 1]
    next_s, next_legal_moves, next_player, done = env.step(s, 1<<9)
    next_s, next_legal_moves = convert_boards(next_s, next_legal_moves)
    correct_s, correct_legal_moves = \
            convert_boards([9151597021616996608, 35183268265393788, 1], 0)
    correct_player = 1
    # do the comparisons stepwise
    success = success & compare_boards([next_s, next_legal_moves, next_player, done], 
                   [correct_s, \
                        correct_legal_moves, correct_player, 1], 
                    5)

    ######## Case 6: custom board from a match ########
    # same match as above
    print('Running Case 6: Custom board from a match')
    s = [4611712458367893760, 3762864076749585532, 0]
    next_s, next_legal_moves, next_player, done = env.step(s, 1<<54)
    next_s, next_legal_moves = convert_boards(next_s, next_legal_moves)
    correct_s, correct_legal_moves = \
            convert_boards([4629797501573202176, 3762793432053759100, 1], 
                               9260674081223606272)
    correct_player = 1
    # do the comparisons stepwise
    success = success & compare_boards([next_s, next_legal_moves, next_player, done], 
                   [correct_s, \
                        correct_legal_moves, correct_player, 0], 
                    6)

    ######## Case 7: custom board from a match ########
    # same match as above
    print('Running Case 7: Custom board from a match')
    s = [9055374549248, 827328404604, 1]
    next_s, next_legal_moves, next_player, done = env.step(s, 1<<46)
    next_s, next_legal_moves = convert_boards(next_s, next_legal_moves)
    correct_s, correct_legal_moves = \
            convert_boards([8917667160320, 71333779971196, 0], 
                               18067175084392960)
    correct_player = 0
    # do the comparisons stepwise
    success = success & compare_boards([next_s, next_legal_moves, next_player, done], 
                   [correct_s, \
                        correct_legal_moves, correct_player, 0], 
                    7)
    
    ######## Case 8: custom board from a match ########
    # https://www.worldothello.org/about/world-othello-championship/woc/2017/round/5
    # Niklas Wettergren 31-33 Brian Rose, moves 40 - 41 
    print('Running Case 8: Custom board from a match')
    s = [8847673277496, 4340405112184963328, 0]
    next_s, next_legal_moves, next_player, done = env.step(s, 1<<54)
    next_s, next_legal_moves = convert_boards(next_s, next_legal_moves)
    correct_s, correct_legal_moves = \
            convert_boards([18058499408542776, 4340369858959180032, 1], 
                               36239903259934790)
    correct_player = 1
    # do the comparisons stepwise
    success = success & compare_boards([next_s, next_legal_moves, next_player, done], 
                   [correct_s, \
                        correct_legal_moves, correct_player, 0], 
                    7)
    # final print
    if(success):
        print('Passed all test cases ! Congrats !')
    else:
        print('One or more test cases failed, correct code and try again !')
    # break from the while loop
    break


# check time taken to play 10000 games
if(success):
    total_games = 10000
    time_list = np.zeros(total_games)
    for i in tqdm(range(total_games)):
        start_time = time.time()
        g.reset()
        winner = g.play()
        time_list[i] = time.time() - start_time
    # print results
    print('Total time taken to play {:d} games : {:.5f}s'.format(total_games, time_list.sum()))
    print('Average time per game : {:.5f}s +- {:.5f}s'.format(np.mean(time_list), np.std(time_list)))


# record and save game
# for i in range(6, 11):
    # g.record_gameplay('images/gameplay_random_{:d}.mp4'.format(i))


# test cases to check if board augmentation is implemented correctly
# first a block of code to check which transformations are redundant
while(0):
    x = np.arange(64, dtype=np.uint8).reshape(-1, 8)
    transition_dict  = {
        'normal'        : x,
        'vertical'      : np.flipud(x).copy(),
        'horizontal'    : np.fliplr(x).copy(),
        'diagonal'      : x.T.copy(),
    }
    base_transitions = list(transition_dict.keys())
    for k in base_transitions:
        x = transition_dict[k]
        transition_dict[k + '_rot_clock_90']     = np.rot90(np.rot90(np.rot90(x))).copy()
        transition_dict[k + '_rot_180']          = np.rot90(np.rot90(x)).copy()
        transition_dict[k + '_rot_anticlock_90'] = np.rot90(x).copy()

    base_transitions = [
        'normal',
        'normal_rot_clock_90',
        'normal_rot_180',
        'normal_rot_anticlock_90',
        'vertical',
        'vertical_rot_clock_90',
        'vertical_rot_180',
        'vertical_rot_anticlock_90',
        'horizontal',
        'horizontal_rot_clock_90',
        'horizontal_rot_180',
        'horizontal_rot_anticlock_90',
        'diagonal',
        'diagonal_rot_clock_90',
        'diagonal_rot_180',
        'diagonal_rot_anticlock_90',
    ]

    remove_list = []
    for i1 in range(len(base_transitions) - 1):
        for i2 in range(i1+1, len(base_transitions)):
            if((transition_dict[base_transitions[i1]] == \
                transition_dict[base_transitions[i2]]).sum() == 64):
                print('{:30s} {:20s}'.format(base_transitions[i1], base_transitions[i2]))
                remove_list.append(base_transitions[i2])

    base_transitions = [x for x in base_transitions if x not in remove_list]
    print('transitions to keep', base_transitions, len(base_transitions))
    """
    transitions to keep ['normal', 'normal_rot_clock_90', 'normal_rot_180', 
    'normal_rot_anticlock_90', 'vertical', 'vertical_rot_clock_90', 'vertical_rot_180', 
    'vertical_rot_anticlock_90'] 8
    """

def get_board_augmentations(transition):
    """
    get the list of all unique transitions obtained through flipping, rotations etc
    considering only anticlockwise rotations, the transitions are normal, normal rotation 270
    normal rotation 180, normal rotation 90, vertical flip, vertical flip rotation 270, 
    vertical flip rotation 180 and vertical flip rotation 90 

    Parameters
    ----------
    transition : list
        contains [[black bitboard, white bitboard, current player], legal moves, current player,
                    action, [next black bitboard, next white bitboard, next player],
                    next legal moves, next player, done, winner]
    
    Returns
    -------
    transition_list : list
        list of augmented transitions
    """
    transition_list = [transition]
    f = lambda x: conv.convert(conv.convert(x, 
                     input_format='bitboard_single', output_format='ndarray'),
                    input_format='ndarray', output_format='bitboard_single')
    for f1 in [lambda x: x, np.flipud]:
        for f2 in [lambda x: x,
                   lambda x: np.rot90(np.rot90(np.rot90(x))),
                   lambda x: np.rot90(np.rot90(x)),
                   lambda x: np.rot90(x)]:
            f_temp = lambda x: f2(f1(f(x)))
            transition_list.append([[f_temp(transition[0][0]), f_temp(transition[0][1]), f_temp(transition[0][2])],
                                       f_temp(transition[1]), transition[2], f_temp(transition[3]),
                                       [f_temp(transition[4][0]), f_temp(transition[4][1]), f_temp(transition[4][2])], 
                                       f_temp(transition[5]), transition[6], transition[7], transition[8]])
    return transition_list

# test cases
while(1):
    print('RUNNING TEST CASES FOR BOARD AUGMENTATION FUNCTION')
    success = True
    env = StateEnvBitBoard(board_size=board_size)
    conv = StateConverter()  
    augmentation = ['normal', 'normal rot 270', 'normal rot 180', 'normal rot 90',
                    'vertical flip', 'vertical flip rot 270', 'vertical flip rot 180', 'vertical flip rot 90']
    ######## Case 1: custom board from a match ########
    # https://www.worldothello.org/about/world-othello-championship/woc/2017/round/5
    # Maria Serena Vecchi 13-51 Caroline Nicolas, move 53 - 54
    print('Running Case 1: Custom board from a match')
    base_transition = [[9055374549248, 827328404604, 1], 8011054621671425, 1, 1<<46, \
                    [8917667160320, 71333779971196, 0], 18067175084392960, 0, 0, 0]
    correct_transitions = get_board_augmentations(base_transition)
    augmented_transitions = g.create_board_reps(base_transition)
    for i in range(len(correct_transitions)):
        if(correct_transitions[i] != augmented_transitions[i]):
            success = False
            print('Case {:d} Augmentation {:s} does not match'.format(1, augmentation[i]))
            print('Expected')
            print(correct_transitions[i])
            print('Got')
            print(augmented_transitions[i])
    
    ######## Case 2: custom board from a match ########
    # https://www.worldothello.org/about/world-othello-championship/woc/2017/round/5
    # Niklas Wettergren 31-33 Brian Rose, moves 40 - 41 
    print('Running Case 2: Custom board from a match')
    base_transition = [[8847673277496, 4340405112184963328, 0], 4774871685931205120, 0, 1<<54, \
                    [18058499408542776, 4340369858959180032, 1], 36239903259934790, 0, 0, 0]
    correct_transitions = get_board_augmentations(base_transition)
    augmented_transitions = g.create_board_reps(base_transition)
    for i in range(len(correct_transitions)):
        if(correct_transitions[i] != augmented_transitions[i]):
            success = False
            print('Case {:d} Augmentation {:s} does not match'.format(1, augmentation[i]))
            print('Expected')
            print(correct_transitions[i])
            print('Got')
            print(augmented_transitions[i])
    
    if(success):
        print('Passed all test cases for board augmentation ! Congrats !')
    else:
        print('One or more test cases failed for board augmentation, correct code and try again !')
    # break from the while loop
    break
