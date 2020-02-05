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
        legal_moves, s = self._legal_moves_helper(starter_board)
        
        return s, legal_moves

    def step(self, s, a):
        """Play a move on the board at the given action location
        and also check for terminal cases, no moves possible etc

        Parameters
        ----------
        s : Ndarray
            current board state
        a : int
            the position where to play the move

        Returns
        -------
        s_next : Ndarray
            updated board
        legal_moves : legal_moves for the next player
        """
        _, s_next = self._legal_moves_helper(s, 
                           action_row=a//self._size, action_col=a%self._size, play=True)
        legal_moves, s_next = self._legal_moves_helper(s_next)
        # return
        return s_next, legal_moves

    def print(self, s):
        """Pretty prints the current board

        Arguments
        ---------
        s : Ndarray
            current board state
        """
        s_print = s[:,:,0] * 10 + s[:,:,1]
        print(s_print)

    def _check_legal_index(self, row, col):
        """Check if the row and col indices are possible in the
        current board

        Parameters
        ----------
        row : int
            row index to check
        col : int
            col index to check

        Returns
        -------
        bool : bool
            if the current indices are out of bound
        """
        return 0 <= row and row < self._size and\
               0 <= col and col < self._size

    def _legal_moves_helper(self, s, action_row=None, action_col=None, play=False):
        """A helper function which iterates over all positions on the board
        to check if the position is a legal move or not. Also, if a particular
        action row and action col are specified, this function, instead of checking
        for legal positions, will modify the board according to the game rules
        and return the modified board. If play, the provided action row and col
        are assumed to be legal

        Parameters
        ----------
        s : Ndarray
            current board state
        action_row : int, default None
            the row in which to play a given move
        action_col : int, default None
            the column in which to play a given move

        Returns
        -------
        available_pos : Ndarray
            the mask containing legal moves (all zeros in case a move
            is to be played)
        s : Ndarray
            the modified board state if play is False
        """
        current_player = s[0,0,2]
        opposite_player = 0 if current_player else 1
        # initialize the array of _size to mark available positions
        legal_moves = np.zeros((self._size, self._size), dtype=np.uint8)

        # determine the loop ranges
        row_range = [action_row] if play else range(self._size)
        col_range = [action_col] if play else range(self._size)

        # modify a copy of the board
        s_new = s.copy()

        # loop over all positions to determine if move is legal
        for row in row_range:
            for col in col_range:
                # check if cell is empty
                if(s[row,col,0] + s[row,col,1] == 0):
                    # modify this position
                    if(play):
                        s_new[row, col, current_player] = 1
                    # check the 8 directions for legal moves/modifying position
                    for del_row, del_col in [[-1,-1], [-1,0], [-1,1],
                                             [0,-1], [0,1],
                                             [1,-1], [1,0], [1,1]]:
                        # check if the index is valid
                        n_row, n_col = row+del_row, col+del_col
                        if(self._check_legal_index(n_row, n_col)):
                            # check if the adjacent cell is of the opposite color
                            if(s[n_row, n_col, opposite_player] == 1):
                                # check if moving in this direction continuously will
                                # lead to coin of same color as current player
                                i = 1
                                found = False
                                while(True):
                                    i += 1
                                    n_row, n_col = row+i*del_row, col+i*del_col
                                    if(self._check_legal_index(n_row, n_col)):
                                        if(s[n_row, n_col, current_player] == 1):
                                            found = True
                                            break
                                    else:
                                        # we have reached terminal position
                                        break
                                if(found):
                                    # the position is valid, modify on the board
                                    legal_moves[row, col] = 1
                                    # modify the respective positions
                                    if(play):
                                        i = -1
                                        while(True):
                                            i += 1
                                            if(row+i*del_row == n_row and col+i*del_col == n_col):
                                                break
                                            s_new[row+i*del_row, col+i*del_col, opposite_player] = 0
                                            s_new[row+i*del_row, col+i*del_col, current_player] = 1
        # change the player
        if(play):
            s_new[:,:,2] = opposite_player
        # return the updated boards
        return legal_moves.copy(), s_new.copy()
