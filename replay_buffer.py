import numpy as np

class ReplayBufferNumpy:
    """This class stores the replay buffer from which data can be sampled for
    training the model for reinforcement learning. Numpy array is used as the
    buffer in this case as it is easier to add multiple steps at once, and 
    sampling is also faster. This is best utilised when using the Numpy array
    based game env

    Attributes
    ----------
    _s : Numpy array
        Buffer for storing the current states, 
        buffer size * board size * board size * 3
    _next_s : Numpy array
        Buffer for storing the next states, 
        buffer size * board size * board size * 3
    _a : Numpy array
        Buffer to store the actions, buffer size * 1
    _done : Numpy array
        Buffer to store the binary indicator for termination
        buffer size * 1
    _r : Numpy array
        Buffer to store the rewards, buffer size * 1
    _next_legal_moves : Numpy array
        Buffer to store the legal moves in the next state, useful
        when calculating the max of Q values in next state
    _buffer_size : int
        Maximum size of the buffer
    _current_buffer_size : int
        Current buffer size, can be used to see if buffer is full
    _pos : int
        Position corresponding to where the next batch of data is
        to be added to the buffer
    _n_actions : int
        Available actions in the env
    """
    def __init__(self, buffer_size=10000, board_size=8, actions=64):
        """Initializes the buffer with given size and also sets attributes

        Parameters
        ----------
        buffer_size : int, optional
            The size of the buffer
        board_size : int, optional
            Board size of the env
        frames : int, optional
            Number of frames used in each state in env
        actions : int, optional
            Number of actions available in env
        """
        self._buffer_size = buffer_size
        self._current_buffer_size = 0
        self._pos = 0
        self._n_actions = actions

        self._s = np.zeros((buffer_size, 3), dtype=np.uint64)
        self._next_s = self._s.copy()
        self._a = np.zeros((buffer_size, 1), dtype=np.uint64)
        self._done = self._a.copy().astype(np.uint8)
        self._r = np.zeros((buffer_size, 1), dtype=np.float32)
        self._next_legal_moves = self._a.copy()

    def add_to_buffer(self, s, a, r, next_s, done, legal_moves):
        """Add data to the buffer, multiple examples can be added at once
        
        Parameters
        ----------
        s : Numpy array
            Current board state, should be a single state
        a : int
            Current action taken
        r : int
            Reward obtained by taking the action on state
        next_s : Numpy array
            Board state obtained after taking the action
            should be a single state
        done : int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicator for legal moves in the next state
        """
        
        l = s.shape[0]
        # % is to wrap over the buffer
        idx = np.arange(self._pos, self._pos+l)%self._buffer_size
        self._s[idx] = s
        self._a[idx] = a
        self._r[idx] = r
        self._next_s[idx] = next_s
        self._done[idx] = done
        self._next_legal_moves[idx] = legal_moves
        # % is to wrap over the buffer
        self._pos = (self._pos+l)%self._buffer_size
        # update the buffer size
        # self._current_buffer_size = max(self._current_buffer_size, self._pos+1)
        if(self._pos + 1 > self._current_buffer_size):
            self._current_buffer_size = self._pos + 1
        else:
            self._current_buffer_size = self._buffer_size

    def get_current_size(self):
        """Returns current buffer size, not to be confused with
        the maximum size of the buffer

        Returns
        -------
        length : int
            Current buffer size
        """
        return self._current_buffer_size

    def sample(self, size=1000, replace=False, shuffle=False):
        """Sample data from buffer and return in easily ingestible form
        returned data has already been reshaped for direct use in the 
        training routine

        Parameters
        ----------
        size : int, optional
            The number of samples to return from the buffer
        replace : bool, optional
            Whether sampling is done with replacement
        shuffle : bool, optional
            Redundant here as the index are already shuffled

        Returns
        -------
        s : Numpy array
            The state matrix for input, size * board size * board size * 3
        a : Numpy array
            Array of actions taken in one hot encoded format, size * num actions
        r : Numpy array
            Array of rewards, size * 1
        next_s : Numpy array
            The next state matrix for input
            The state matrix for input, size * board size * board size * 3
        done : Numpy array
            Binary indicators for game termination, size * 1
        legal_moves : Numpy array
            Binary indicators for legal moves in the next state, size * num actions
        """
        size = min(size, self._current_buffer_size)
        # select random indexes indicating which examples to sample
        idx = np.random.choice(np.arange(self._current_buffer_size), \
                                    size=size, replace=replace)

        s = self._s[idx]
        a = self._a[idx]
        r = self._r[idx].reshape((-1, 1))
        next_s = self._next_s[idx]
        done = self._done[idx].reshape(-1, 1)
        next_legal_moves = self._next_legal_moves[idx]

        return s, a, r, next_s, done, next_legal_moves
