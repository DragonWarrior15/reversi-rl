"""Stores classes for the game environment
and the class to play the game between two players
"""

import numpy as np
import os

class state_env:
    """Base class that implements all the rules of the game
    and also provides the public functions that an agent can
    use to play the game. For determining which player will play,
    white is represented by 1 and black by 0. Note that this class
    has no memory of the game state, and will only perform the task
    of initializing the board, and given an existing state and action,
    perform that action and return the new state, next turn etc.
    The implementation of the game history will be done in a separate
    game class with additional functionality.
    
    Attributes
    ----------
    _size : int
        the size of the board to be played on
    """
    def __init__(self, board_size=8):
        """Initializer for the game environment
        
        Parameters
        ----------
        board_size : int
            kept 8 as the default for othello
        """
        self._size = board_size if board_size%2 == 0 else board_size+1
        # a list of helper convolutions
        self._blank_pos_conv = np.zeros((3,3,9), dtype=np.uint8)
        # fill 1 individually
        temp = np.arange(9).reshape(3, 3)
        for i in range(9):
            if(i == 4):
                continue
            k = i if i<4 else i-1
            self._blank_pos_conv[:,:,i if i<4 else i-1][i == temp] = 2**k

    def reset(self):
        """Reset the board to starting configuration
        any handicap configurations will go here
        
        Returns
        -------
        starter_board : Ndarray
            the numpy array containing the starting configuration
            of the othello board, positions of black coin are at the
            0th index, white at 1st index and current player at 2nd index
        legal_moves : Ndarray
            masked array denoting the legal positions
        """
        starter_board = np.zeros((self._size, self._size, 3), 
                                       dtype=np.uint8)
        # put the starting coins
        half_width = (self._size//2) - 1
        # put white coins
        starter_board[[half_width, half_width+1], \
                                    [half_width, half_width+1], 1] = 1
        # put black coins
        starter_board[[half_width, half_width+1], \
                                    [half_width+1, half_width], 0] = 1
        # white to start
        starter_board[:, :, 2] = 1
        # get legal moves
        legal_moves = self._get_legal_moves(starter_board)
        
        return starter_board.copy(), legal_moves.copy()

    def _get_legal_moves(self, s):
        """Returns the legal moves for current board state
        
        Parameters
        ----------
        s : Ndarray
            the current board state
        
        Returns
        -------
        legal_moves : Ndarray
            masked array with legal positions marked with a 1
        """
        # first mask the available positions for current player in adjacent positions
        # to the coins of the opposite player
        current_player = s[0,0,2]
        # initialize the array of _size+2 since convolutions reduce
        # the size of input array by 2
        available_pos = np.zeros((self._size+2, self._size+2), dtype=np.uint8)
        # we have to check adjacent
        available_pos[1:self._size+1, 1:self._size+1] = s[:,:,0 if current_player else 1]
        pstr = available_pos.strides
        # create the exploded view for convolutions
        available_pos = np.lib.stride_tricks.as_strided(available_pos, 
                       shape=(self._size,self._size,3,3),
                       strides=(pstr[0],pstr[1],pstr[0],pstr[1]),
                       writeable=False)
        # do the convolution operation (with dot product)
        available_pos = np.tensordot(available_pos, self._blank_pos_conv).sum(2)
        # clip to make the max value at identified position 1
        available_pos = np.clip(available_pos, 0, 1)
        # postions with coins already present should not be counted
        available_pos[s[:,:,:2].sum(2) == 1] = 0

        return available_pos




