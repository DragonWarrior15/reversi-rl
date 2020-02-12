"""This module holds all the players that can interact with the othello board"""
import numpy as np

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
            legal states are set

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