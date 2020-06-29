"""This module holds all the players that can interact with the othello board"""
import numpy as np
from game_env import StateEnvBitBoard

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
        idx = 0
        move_list = []
        while(legal_moves):
            if(legal_moves & 1):
                move_list.append(idx)
            legal_moves = legal_moves >> 1
            idx += 1
        np.random.shuffle(move_list)
        # idx represents position from end
        # hence bitboard can be prepared by simply shifting 1
        # by the idx
        return 1 << move_list[0]

class MiniMaxPlayer(Player):
    """This agent uses the minimiax algorithm to decide which move to play

    Attributes
    _depth : int
        the depth to which explore the next states for best move
    _env : StateEnvBitBoard
        instance of environment to allow exploration of next states
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
        self._depth = max(depth, 1)
        # an instance of the environment
        self._env = StateEnvBitBoard(board_size=board_size)

    def move(self, s, legal_moves, current_depth=0, get_max=True):
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
        get_max : bool
            denotes whether to play as maximum/original player, 
            only useful when recursion depth > 1

        Returns
        -------
        a : int (64 bit)
            bitboard representing position to play
        """
        # max player
        max_player = s[2]
        # get the indices of the legal moves
        move_list = []
        idx = 0
        while(legal_moves):
            if(legal_moves & 1):
                move_list.append(idx)
            legal_moves = legal_moves >> 1
            idx += 1
        h_list = []
        for m in move_list:
            s_next, legal_moves, _, done = self._env.step(s, 1 << m)
            if(current_depth < self._depth and not done):
                h_list.append(self.move(s_next, legal_moves, 
                                        current_depth+1, ~get_max))
            else:
                h_list.append(self._board_heuristics(s_next))
        # return the best move
        if(current_depth == 0):
            return 1 << move_list[np.argmax(h_list)]
        if(get_max):
            return np.max(h_list)
        else:
            return np.min(h_list)
        

    def _board_heuristics(self, s):
        """Get a number representing the goodness of the board state

        Parameters
        ----------
        s : tuple
            contains the bitboards for the current game state

        Returns
        -------
        h : int
            an int denoting how good the board is for the current player
        """
        b, w = self._env.count_coins(s)
        h = b - w if self._env.get_player(s) == 'black' else w - b
        return h
