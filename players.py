"""This module holds all the players that can interact with the othello board"""
import numpy as np
from game_env import (StateEnvBitBoard, get_set_bits_list,
                get_total_set_bits, get_random_move_from_list,
                StateEnvBitBoardC)
from mcts import MCTS
import ctypes
import time
import pickle
from collections import deque
import json
from replay_buffer import ReplayBufferNumpy
# import tensorflow as tf
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.optimizers import RMSprop, SGD, Adam
# from keras.losses import Loss
# import tensorflow.keras.backend as K
# from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Softmax, MaxPool2D
# from tensorflow.keras import Model
# from tensorflow.keras.regularizers import l2

def huber_loss(y_true, y_pred, delta=1):
    """Keras implementation for huber loss
    loss = {
        0.5 * (y_true - y_pred)**2 if abs(y_true - y_pred) < delta
        delta * (abs(y_true - y_pred) - 0.5 * delta) otherwise
    }
    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        loss values for all points
    """
    error = (y_true - y_pred)
    quad_error = 0.5*tf.math.square(error)
    lin_error = delta*(tf.math.abs(error) - 0.5*delta)
    # quadratic error, linear error
    return tf.where(tf.math.abs(error) < delta, quad_error, lin_error)

def mean_huber_loss(y_true, y_pred, delta=1):
    """Calculates the mean value of huber loss

    Parameters
    ----------
    y_true : Tensor
        The true values for the regression data
    y_pred : Tensor
        The predicted values for the regression data
    delta : float, optional
        The cutoff to decide whether to use quadratic or linear loss

    Returns
    -------
    loss : Tensor
        average loss across points
    """
    return tf.reduce_mean(huber_loss(y_true, y_pred, delta))

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

    def __init__(self, board_size=8):
        Player.__init__(self, board_size)
        self.name = 'Random'


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

class DeepQLearningAgent():
    """This agent learns the game via Q learning
    model outputs everywhere refers to Q values
    This class can be extended to the following classes
    PolicyGradientAgent
    AdvantageActorCriticAgent

    Attributes
    ----------
    _board_size : int
        Size of board, keep greater than 6 for useful learning
        should be the same as the env board size
    _buffer_size : int
        Size of the buffer, how many examples to keep in memory
        should be large for DQN
    _n_actions : int
        Total actions available in the env, should be same as env
    _gamma : float
        Reward discounting to use for future rewards, useful in policy
        gradient, keep < 1 for convergence
    _use_target_net : bool
        If use a target network to calculate next state Q values,
        necessary to stabilise DQN learning
    _input_shape : tuple
        Tuple to store individual state shapes
    _version : str
        model version string
    _model : TensorFlow Graph
        Stores the graph of the DQN model
    _s_input_format : str
        input format in which player accepts board state 
    _legal_moves_input_format : str
        input format in which player accepts legal moves mask
    _move_output_format : str
        output format in which player returns the selected move
    name : str
        Name of the agent
    epsilon: float, optional
        Agent's exploration probability, necessary for DQN convergence
    """

    def __init__(self, board_size=8, buffer_size=10000,
                 gamma=0.99, n_actions=64, use_target_net=True,
                 epsilon=0.9, version=''):
        """initialize the agent

        Parameters
        ----------
        board_size : int, optional
            The env board size, keep > 6
        buffer_size : int, optional
            Size of the buffer, keep large for DQN
        gamma : float, optional
            Agent's discount factor, keep < 1 for convergence
        n_actions : int, optional
            Count of actions available in env
        use_target_net : bool, optional
            Whether to use target network, necessary for DQN convergence
        epsilon: float, optional
            Agent's exploration probability, necessary for DQN convergence
        version : str, optional except NN based models
            path to the model architecture json
        """
        self._board_size = board_size
        self._buffer_size = buffer_size
        self._n_actions = n_actions
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._input_shape = (self._board_size, self._board_size, 3)
        self._converter = StateConverter()
        # reset buffer also initializes the buffer
        self.reset_buffer()
        self._version = version
        # io format for state (not relevant here)
        self._s_input_format = 'ndarray3d'
        # io format for legal moves
        self._legal_moves_input_format = 'bitboard_single'
        # io for output (move)
        self._move_output_format = 'bitboard_single'
        self.epsilon = epsilon
        self.name = 'DQN'
        self.reset_models()

    def get_state_input_format(self):
        """Return the input format for game state"""
        return self._s_input_format

    def get_legal_moves_input_format(self):
        """Return the input format for legal moves"""
        return self._legal_moves_input_format

    def get_move_output_format(self):
        """Return the output format for selected move"""
        return self._move_output_format

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer 
        
        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if(buffer_size is not None):
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(self._buffer_size, self._board_size, 
                                         self._n_actions)

    def get_buffer_size(self):
        """Get the current buffer size
        
        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, 
                                   done, legal_moves)

    def save_buffer(self, file_path='', iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'wb') as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path='', iteration=None):
        """Load the buffer from disk
        
        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), 'rb') as f:
            self._buffer = pickle.load(f)

    def reset_models(self):
        """ Reset all the models by creating new graphs"""
        self._model = self._agent_model()
        if(self._use_target_net):
            self._target_net = self._agent_model()
            self.update_target_net()

    def _prepare_input(self, board):
        """Reshape input and normalize
        
        Parameters
        ----------
        board : Numpy array
            The board state to process

        Returns
        -------
        board : Numpy array
            Processed and normalized board
        """
        if(board.ndim == 3):
            board = board.reshape((1,) + self._input_shape)
        return board.copy()

    def _get_model_outputs(self, board, model=None):
        """Get action values from the DQN model

        Parameters
        ----------
        board : Numpy array
            The board state for which to predict action values
        model : TensorFlow Graph, optional
            The graph to use for prediction, model or target network

        Returns
        -------
        model_outputs : Numpy array
            Predicted model outputs on board, 
            of shape board.shape[0] * num actions
        """
        # to correct dimensions and normalize
        board = self._prepare_input(board)
        # the default model to use
        if model is None:
            model = self._model
        model_outputs = model.predict_on_batch(board)
        return model_outputs

    def move(self, board, legal_moves, value=None):
        """Get the action with maximum Q value
        
        Parameters
        ----------
        board : Numpy array
            The board state on which to calculate best action
        value : None, optional
            Kept for consistency with other agent classes

        Returns
        -------
        output : Numpy array
            Selected action using the argmax function
        """
        # use the agent model to make the predictions
        if np.random.random()  > self.epsilon:
            model_outputs = self._get_model_outputs(board, self._model)[0]
            legal_moves = self._converter.convert(legal_moves, input_format='bitboard_single',\
                                                  output_format='ndarray')\
                              .reshape((1, -1))[0]
            return 1 << int((63 - np.argmax(np.where(legal_moves==1, model_outputs, -np.inf))))

        else:
            if(not legal_moves):
                a = 0
            return 1 << get_random_move_from_list(get_set_bits_list(legal_moves))

    def _agent_model(self):
        """Returns the model which evaluates Q values for a given state input

        Returns
        -------
        model : TensorFlow Graph
            DQN model graph
        """
        # define the input layer, shape is dependent on the board size and frames
        with open('model_config/{:s}.json'.format(self._version), 'r') as f:
            m = json.loads(f.read())
        
        input_board = Input((self._board_size, self._board_size, 3), name='input')
        x = input_board
        for layer in m['model']:
            l = m['model'][layer]
            if('Conv2D' in layer):
                # add convolutional layer
                x = Conv2D(**l)(x)
            if('Flatten' in layer):
                x = Flatten()(x)
            if('Dense' in layer):
                x = Dense(**l)(x)
        out = Dense(self._n_actions, activation='linear', name='action_values')(x)
        model = Model(inputs=input_board, outputs=out)
        model.compile(optimizer=RMSprop(0.0005), loss=mean_huber_loss)
                
        """
        input_board = Input((self._board_size, self._board_size, 3,), name='input')
        x = Conv2D(16, (3,3), activation='relu', data_format='channels_last')(input_board)
        x = Conv2D(32, (3,3), activation='relu', data_format='channels_last')(x)
        x = Conv2D(64, (4,4), activation='relu', data_format='channels_last')(x)
        x = Flatten()(x)
        x = Dense(64, activation = 'relu', name='action_prev_dense')(x)
        # this layer contains the final output values, activation is linear since
        # the loss used is huber or mse
        out = Dense(self._n_actions, activation='linear', name='action_values')(x)
        # compile the model
        model = Model(inputs=input_board, outputs=out)
        model.compile(optimizer=RMSprop(0.0005), loss=mean_huber_loss)
        # model.compile(optimizer=RMSprop(0.0005), loss='mean_squared_error')
        """

        return model

    def set_weights_trainable(self):
        """Set selected layers to non trainable and compile the model"""
        for layer in self._model.layers:
            layer.trainable = False
        # the last dense layers should be trainable
        for s in ['action_prev_dense', 'action_values']:
            self._model.get_layer(s).trainable = True
        self._model.compile(optimizer = self._model.optimizer, 
                            loss = self._model.loss)


    def get_action_proba(self, board, values=None):
        """Returns the action probability values using the DQN model

        Parameters
        ----------
        board : Numpy array
            Board state on which to calculate action probabilities
        values : None, optional
            Kept for consistency with other agent classes
        
        Returns
        -------
        model_outputs : Numpy array
            Action probabilities, shape is board.shape[0] * n_actions
        """
        model_outputs = self._get_model_outputs(board, self._model)
        # subtracting max and taking softmax does not change output
        # do this for numerical stability
        model_outputs = np.clip(model_outputs, -10, 10)
        model_outputs = model_outputs - model_outputs.max(axis=1).reshape((-1,1))
        model_outputs = np.exp(model_outputs)
        model_outputs = model_outputs/model_outputs.sum(axis=1).reshape((-1,1))
        return model_outputs

    def save_model(self, file_path='', iteration=None):
        """Save the current models to disk using tensorflow's
        inbuilt save model function (saves in h5 format)
        saving weights instead of model as cannot load compiled
        model with any kind of custom object (loss or metric)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.save_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if(self._use_target_net):
            self._target_net.save_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))

    def load_model(self, file_path='', iteration=None):
        """ load any existing models, if available """
        """Load models from disk using tensorflow's
        inbuilt load model function (model saved in h5 format)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if(iteration is not None):
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        self._model.load_weights("{}/model_{:04d}.h5".format(file_path, iteration))
        if(self._use_target_net):
            self._target_net.load_weights("{}/model_{:04d}_target.h5".format(file_path, iteration))
        # print("Couldn't locate models at {}, check provided path".format(file_path))

    def print_models(self):
        """Print the current models using summary method"""
        print('Training Model')
        print(self._model.summary())
        if(self._use_target_net):
            print('Target Network')
            print(self._target_net.summary())

    def convert_bitboards(self, s, a, next_s, legal_moves):
        """ Converts states, rewards, actions, next states and legal moves sample
        from agent's buffer into numpy arrays of desired shape for training the model

        Parameters
        ----------
        s: Numpy array of shape (self._batch_size, 3)
        a: Numpy array of shape (self._batch_size, 1)
        next_s: Numpy array of shape (self._batch_size, 3)
        legal_moves: Nunpy array of shape (self._batch_size, 1)

        Returns
        -------
        s_board: Numpy array of shape (self._batch_size, self._board_size, self._board_size, 3)
        a_board: Numpy array of shape (self._batch_size, self._n_actions)
        next_s: Numpy array of shape (self._batch_size, self._board_size, self._board_size, 3)
        legal_moves: Numpy array of shape (self._batch_size, self._n_actions)

        """

        s_board = np.zeros((len(s), self._board_size, self._board_size, 3), dtype='uint8')
        next_s_board = s_board.copy()
        a_board = np.zeros((len(s), self._n_actions), dtype='uint8')
        legal_moves_board = a_board.copy()
        for i in range(len(s)):
            s_board[i] = self._converter.convert([int(item) for item in list(s[i])],\
                                                 input_format='bitboard',\
                                                 output_format='ndarray3d')
            next_s_board[i] = self._converter.convert([int(item) for item in list(next_s[i])],\
                                                      input_format='bitboard',\
                                                      output_format='ndarray3d')
            a_board[i] = self._converter.convert(int(a[i][0]),\
                                                 input_format='bitboard_single',\
                                                 output_format='ndarray').reshape(-1, self._n_actions)
            legal_moves_board[i] = self._converter.convert(int(legal_moves[i][0]),\
                                                           input_format='bitboard_single',\
                                                           output_format='ndarray').reshape(-1, self._n_actions)

        return s_board, a_board, next_s_board, legal_moves_board


    def train_agent(self, batch_size=32, num_games=1, reward_clip=False):
        """Train the model by sampling from buffer and return the error.
        We are predicting the expected future discounted reward for all
        actions with our model. The target for training the model is calculated
        in two parts:
        1) dicounted reward = current reward + 
                        (max possible reward in next state) * gamma
           the next reward component is calculated using the predictions
           of the target network (for stability)
        2) rewards for only the action take are compared, hence while
           calculating the target, set target value for all other actions
           the same as the model predictions
        
        Parameters
        ----------
        batch_size : int, optional
            The number of examples to sample from buffer, should be small
        num_games : int, optional
            Not used here, kept for consistency with other agents
        reward_clip : bool, optional
            Whether to clip the rewards using the numpy sign command
            rewards > 0 -> 1, rewards <0 -> -1, rewards == 0 remain same
            this setting can alter the learned behaviour of the agent

        Returns
        -------
            loss : float
            The current error (error metric is defined in reset_models)
        """
        s, a, r, next_s, done, legal_moves = self._buffer.sample(batch_size)
        # converting states, actions and moves from bitboards to numpy arrays
        s_board, a_board, next_s_board, legal_moves_board = self.convert_bitboards(s, a, next_s, legal_moves)
        if(reward_clip):
            r = np.sign(r)
        # calculate the discounted reward, and then train accordingly
        current_model = self._target_net if self._use_target_net else self._model
        next_model_outputs = self._get_model_outputs(next_s_board, current_model)
        # our estimate of expexted future discounted reward
        discounted_reward = r + \
                 (self._gamma * np.max(np.where(legal_moves_board == 1, next_model_outputs, -np.inf), 
                                  axis = 1)\
                                  .reshape(-1, 1)) * (1-done)
                # create the target variable, only the column with action has different value
        target = self._get_model_outputs(s_board)
        # we bother only with the difference in reward estimate at the selected action
        target = (1-a_board)*target + a_board*discounted_reward
        # fit
        loss = self._model.train_on_batch(s_board, target)
        # loss = round(loss, 5)
        return loss

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if(self._use_target_net):
            self._target_net.set_weights(self._model.get_weights())

    def compare_weights(self):
        """Simple utility function to heck if the model and target 
        network have the same weights or not
        """
        for i in range(len(self._model.layers)):
            for j in range(len(self._model.layers[i].weights)):
                c = (self._model.layers[i].weights[j].numpy() == \
                     self._target_net.layers[i].weights[j].numpy()).all()
                print('Layer {:d} Weights {:d} Match : {:d}'.format(i, j, int(c)))

    def copy_weights_from_agent(self, agent_for_copy):
        """Update weights between competing agents which can be used
        in parallel training
        """
        assert isinstance(agent_for_copy, self), "Agent type is required for copy"

        self._model.set_weights(agent_for_copy._model.get_weights())
        self._target_net.set_weights(agent_for_copy._model_pred.get_weights())

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
