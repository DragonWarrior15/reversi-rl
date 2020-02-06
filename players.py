"""This module holds all the players that can interact with the othello board"""
import numpy as np

class Player:
    """The base class for player. All common attributes/functions go here"""
    def __init__(self, board_size=8):
        """Initializer

        Parameters
        ----------
        board_size : int
            size of game board
        """
        self._size = board_size

class RandomPlayer(Player):
    """Random player that selects moves randomly
    from all the legal moves
    """
    def move(self, s, legal_moves):
        """Select a move randomly, given the board state and the
        set of legal moves

        Parameters
        ----------
        s : Ndarray
            board state
        legal_moves : Ndarray
            mask of the legal states

        Returns
        -------
        a : int
            the position where to play
        """
        legal_moves = legal_moves * np.random.rand(*legal_moves.shape)
        legal_moves[legal_moves == legal_moves.max()] = 1
        legal_moves[legal_moves < 1] = 0
        legal_moves = legal_moves * np.arange(self._size**2).reshape(legal_moves.shape)
        return int(legal_moves.max())