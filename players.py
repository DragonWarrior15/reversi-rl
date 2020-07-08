"""This module holds all the players that can interact with the othello board"""
import numpy as np
from game_env import (StateEnvBitBoard, get_set_bits_list,
                get_total_set_bits, get_random_move_from_list,
                StateEnvBitBoardC)
from mcts import MCTS
import ctypes

class Player:
    """The base class for player. All common attributes/functions go here

    Attributes
    _size : int
        the board size
    _s_input_format : str
        input format in which player accepts board state 
    _legal_moves_input_format : str
        input format in which player accepts legal moves mask
    _move_output_format : str
        output format in which player returns the selected move
    """
    def __init__(self, board_size=8):
        """Initializer

        Parameters
        ----------
        board_size : int
            size of game board
        """
        self._size = board_size
        # io format for state (not relevant here)
        self._s_input_format = 'bitboard'
        # io format for legal moves
        self._legal_moves_input_format = 'bitboard_single'
        # io for output (move)
        self._move_output_format = 'bitboard_single'

    def get_state_input_format(self):
        """Return the input format for game state"""
        return self._s_input_format

    def get_legal_moves_input_format(self):
        """Return the input format for legal moves"""
        return self._legal_moves_input_format

    def get_move_output_format(self):
        """Return the output format for selected move"""
        return self._move_output_format

class RandomPlayer(Player):
    """Random player that selects moves randomly
    from all the legal moves
    """
    def move(self, s, legal_moves):
        """Select a move randomly, given the board state and the
        set of legal moves

        Parameters
        ----------
        s : tuple
            contains black and white bitboards and current player
        legal_moves : int (64 bit)
            legal states are set to 1

        Returns
        -------
        a : int (64 bit)
            bitboard representing position to play
        """
        if(not legal_moves):
            return 0
        return 1 << get_random_move_from_list(get_set_bits_list(legal_moves))

class MiniMaxPlayer(Player):
    """This agent uses the minimiax algorithm to decide which move to play

    Attributes
    _depth : int
        the depth to which explore the next states for best move
    _env : StateEnvBitBoard
        instance of environment to allow exploration of next states
    _player : int
        stores which coin the player has to originally play
        1 for white and 0 for black
    """

    def __init__(self, board_size=8, depth=1):
        """Initializer

        Parameters
        ----------
        board_size : int
            size of game board
        depth : int
            the depth to look ahead for best moves, >= 1
        """
        self._size = board_size
        # io format for state (not relevant here)
        self._s_input_format = 'bitboard'
        # io format for legal moves
        self._legal_moves_input_format = 'bitboard_single'
        # io for output (move)
        self._move_output_format = 'bitboard_single'
        # set the depth
        self._depth = max(depth, 0)
        # an instance of the environment
        self._env = StateEnvBitBoardC(board_size=board_size)

    def move(self, s, legal_moves, current_depth=0, get_max=1,
             alpha=-np.inf, beta=np.inf):
        """Select a move randomly, given the board state and the
        set of legal moves

        Parameters
        ----------
        s : tuple
            contains black and white bitboards and current player
        legal_moves : int (64 bit)
            legal states are set to 1
        current_depth : int
            tracks the depth in the recursion
        get_max : int
            denotes whether to play as maximum/original player, 
            only useful when recursion depth > 1, 1 is max and 0 is min player
        alpha : int
            tracks the maximum among all the nodes, useful for pruning
        beta : int
            tracks the minimum among all the nodes, useful for pruning

        Returns
        -------
        a : int (64 bit)
            bitboard representing position to play
        """
        # max player
        if(current_depth == 0):
            self._player = self._env.get_player(s)
        # get the indices of the legal moves
        move_list = get_set_bits_list(legal_moves)
        h_list = []
        m_list = []
        for m in move_list:
            s_next, legal_moves, _, done = self._env.step(s, 1 << m)
            if(current_depth < self._depth and not done):
                h_list.append(self.move(s_next, legal_moves, 
                                        current_depth+1, 1-get_max,
                                        alpha,beta))
            else:
                h_list.append(self._board_heuristics(legal_moves,
                              get_max, s_next))
            m_list.append(m)
            # print(current_depth, h_list, m, legal_moves, s, alpha, beta)
            # adjust alpha and beta
            # print(current_depth, alpha, beta, h_list[-1], 
                  # len(move_list), m, get_max)
            if(get_max):
                alpha = max(alpha, h_list[-1])
            else:
                beta = min(beta, h_list[-1])
            if(beta <= alpha):
                break
        # return the best move
        if(current_depth == 0):
            return 1 << m_list[np.argmax(h_list)]
        if(get_max):
            return alpha
        else:
            return beta
        

    def _board_heuristics(self, legal_moves, get_max, s):
        """Get a number representing the goodness of the board state
        here, we evaluate that by counting how many moves can be played

        Parameters
        ----------
        legal_moves : 64 bit int
            each bit is a flag for whether that position is valid move
        get_max : int
            flag 1 or 0 to determining what output to return
        s : tuple
            contains the state bitboards

        Returns
        -------
        h : int
            an int denoting how good the board is for the current player
        """
        # this function uses the difference in coins in the next state
        b,w = self._env.count_coins(s)
        player = self._env.get_player(s)
        if(player):
            return w - b
        else:
            return b - w

        # this function uses the no of moves avaialable in the next state
        # and might fail later in the game when board is highly occupied
        # if(get_max):
            # return get_total_set_bits(legal_moves)
        # return -get_total_set_bits(legal_moves)

class MiniMaxPlayerC(MiniMaxPlayer):
    """Extends the MiniMaxPlayer to implement a pure C based move 
    generation function"""

    def __init__(self, board_size=8, depth=1):
        """Initializer

        Parameters
        ----------
        board_size : int
            size of game board
        depth : int
            the depth to look ahead for best moves, >= 1
        """
        MiniMaxPlayer.__init__(self, board_size=board_size, depth=depth)
        self._env = ctypes.CDLL('minimax.dll')

    def move(self, s, legal_moves):
        """Select a move randomly, given the board state and the
        set of legal moves

        Parameters
        ----------
        s : tuple
            contains black and white bitboards and current player
        legal_moves : int (64 bit)
            legal states are set to 1
        """
        """for C, alpha and beta cannot be passed as inf as it requires more
        dependencies to be installed, hence we will pass the max and min
        value based on our judgement
        for a heuristic based on number of moves, alpha, beta can be -+64
        for a heuristic on difference of coins
        """
        m = self._env.move(ctypes.c_ulonglong(s[0]), ctypes.c_ulonglong(s[1]), 
              ctypes.c_ulonglong(legal_moves), ctypes.c_uint(0),
              ctypes.c_uint(1), ctypes.c_int(-64), ctypes.c_int(+64), 
              ctypes.c_uint(s[2]), ctypes.c_uint(self._depth), 
              ctypes.c_uint(s[2]))
        return 1 << m


class MCTSPlayer(Player):
    """This agent uses MCTS to decide which move to play"""

    def move(self, s, legal_moves):
        """Select a move randomly, given the board state and the
        set of legal moves

        Parameters
        ----------
        s : tuple
            contains black and white bitboards and current player
        legal_moves : int (64 bit)
            legal states are set to 1

        Returns
        -------
        a : int (64 bit)
            bitboard representing position to play
        """
        # initialize mcts and train it
        mcts = MCTS(s, legal_moves, board_size=self._size)
        mcts.train()
        return mcts.select_move()


class MCTSPlayerC(Player):
    """This agent uses MCTS C implementation to decide which move to play"""
    def __init__(self, board_size=8):
        Player.__init__(self, board_size=board_size)
        self._env = ctypes.CDLL('mcts.dll')

    def move(self, s, legal_moves):
        """Select a move randomly, given the board state and the
        set of legal moves

        Parameters
        ----------
        s : tuple
            contains black and white bitboards and current player
        legal_moves : int (64 bit)
            legal states are set to 1

        Returns
        -------
        m : int (64 bit)
            bitboard representing position to play
        """
        # train mcts and get the move
        m = self._env.move(ctypes.c_ulonglong(s[0]), ctypes.c_ulonglong(s[1]), 
                    ctypes.c_uint(s[2]), ctypes.c_ulonglong(legal_moves), 
                    ctypes.c_uint(100))
        return 1 << m
