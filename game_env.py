"""Stores classes for the game environment
and the class to play the game between two players
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import shutil
from tqdm import tqdm

class StateEnv:
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
        current_player : int
            the integer denoting which player plays
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
        
        return s, legal_moves, 1

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
        legal_moves : Ndarray
            legal_moves for the next player
        current_player : int
            whether 1 to play next or 0
        done : int
            1 if the game terminates, else 0
        """
        # variable to track if the game has ended
        # rewards will be determined by the game class
        done = 0
        _, s_next = self._legal_moves_helper(s, 
                           action_row=a//self._size, action_col=a%self._size, play=True)
        # change the player before checking for legal moves
        s_next[:,:,2] = abs(s[0,0,2]-1)
        legal_moves, s_next = self._legal_moves_helper(s_next)
        # check if legal moves are available
        if(legal_moves.sum() == 0):
            # either the current player cannot play, or the game has ended
            if(s_next[:,:,:2].sum() == self._size**2):
                # game has ended
                done = 1
            else:
                # current player cannot play, switch player
                s_next[:,:,2] = s[0,0,2]
                # check for legal moves again
                legal_moves, _ = self._legal_moves_helper(s_next)
                if(legal_moves.sum() == 0):
                    # no moves are possible, game is over
                    done = 1
                else:
                    # original player will play next and opposite player
                    # will pass the turn, nothing to modify
                    pass
        # return
        return s_next, legal_moves, int(s_next[0, 0, 2]), done

    def print(self, s, legal_moves=None):
        """Pretty prints the current board

        Arguments
        ---------
        s : Ndarray
            current board state
        """
        print(('black 0 ' if s[0,0,2]==0 else 'white 1 ') + 'to play')
        s_print = s[:,:,0] * 10 + s[:,:,1]
        print(s_print)
        if(legal_moves is not None):
            print(legal_moves * np.arange(self._size**2).reshape(-1, self._size))

    def count_coins(self, s):
        """Count the black and white coins on the board.
        Useful to check winner of the game
        
        Parameters
        ----------
        s : Ndarray
            the board state
        
        Returns
        -------
        (b, w) : tuple
            tuple of ints containing the coin counts
        """
        return (s[:,:,0].sum(), s[:,:,1].sum())

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
                                        # if this cell is blank, break
                                        if(s[n_row, n_col, :2].sum() == 0):
                                            break
                                        # if current player cell encountered again
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
                                        i = 0
                                        while(True):
                                            i += 1
                                            if(row+i*del_row == n_row and col+i*del_col == n_col):
                                                break
                                            s_new[row+i*del_row, col+i*del_col, opposite_player] = 0
                                            s_new[row+i*del_row, col+i*del_col, current_player] = 1
        # do not change the player in this function, instead do
        # this in the step function to check for potential end conditions
        # if(play):
            # s_new[:,:,2] = opposite_player
        # return the updated boards
        return legal_moves.copy(), s_new.copy()

class Game:
    """This class handles the complete lifecycle of a game,
    it keeping history of all the board state, keeps track of two players
    and determines winners, rewards etc

    Attributes
    _p1 : Player
        the first player
    _p2 : Player
        the second player
    _size : int
        the board size
    _env : StateEnv
        the state env to play game
    _p : dict
        the mapping indicating color of players
    _hist : list
        stores all the state transitions
        [current board, current legal moves, current player, action,
         next board, next legal moves, next player, done, reward]
    """
    def __init__(self, player1, player2, board_size=8):
        """Initialize the game with the two specified players
        the players have a move function which fetches where to play
        the stone of their color next

        Parameters
        ----------
        player1 : Player
            the first player
        player2 : Player
            the second player
        board_size : int, default 8
            the size of board for othello
        """
        self._p1 = player1
        self._p2 = player2
        self._size = board_size
        self._env = StateEnv(board_size=board_size)
        self._rewards = {'tie':0, 'win':1, 'loss':-1}

    def reset(self, random_assignment=False):
        """Reset the game and randomly select who plays first"""
        # reset environment
        _, _, _ = self._env.reset()
        # prepare object to track history
        self._hist = []
        # assign the first player
        self._p = {1:self._p1, 0:self._p2}
        if(np.random.rand() < 0.5 and random_assignment):
            self._p = {0:self._p1, 1:self._p2}

    def play(self):
        """Play the game to the end"""
        # get the starting state
        s, legal_moves, current_player = self._env.reset()
        done = 0
        while not done:
            # get the action from current player
            a = self._p[current_player].move(s, legal_moves)
            # step the environment
            next_s, next_legal_moves, next_player, done = self._env.step(s, a)
            # add to the historybject
            self._hist.append([s.copy(), legal_moves.copy(), current_player, a, \
                               next_s.copy(), next_legal_moves.copy(), next_player, 0])
            # setup for next iteration of loop
            s = next_s.copy()
            current_player = next_player
            legal_moves = next_legal_moves

        # determine the winner
        b, w = self._env.count_coins(s)
        if(b > w):
            winner = 0
        elif(w > b):
            winner = 1
        else:
            # tie
            winner = -1

        # modify the history object
        self._hist[-1][-1] = winner

        return winner

    def record_gameplay(self, path='file.mp4'):
        """Plays a game and saves the frames as individual pngs
    
        Parameters
        ----------
        path : str
            the file name where to save the mp4/gif
        """
        frames_dir = 'temp_frames'
        # color transition from black to white
        transition_color_list = ['forestgreen', 'black', 'dimgray', 'dimgrey', 'gray', 'grey',
                                 'darkgray', 'darkgrey', 'silver', 'lightgray', 
                                 'lightgrey', 'gainsboro', 'whitesmoke', 'white']
        frames_per_anim = len(transition_color_list) - 1
        color_array = np.zeros((self._size, self._size), np.uint8)
        alpha_array = np.zeros((self._size, self._size), np.uint8)
        # reset the game
        self.reset()
        # play a full game
        winner = self.play()
        # use the history object to save to game
        # temp_frames directory is reserved to save intermediate frames
        if(os.path.exists(frames_dir)):
            # clear out the directory
            # shutil.rmtree(frames_dir)
            for _, _, file_list in os.walk(frames_dir):
                pass
            for f in file_list:
                os.remove(os.path.join(frames_dir, f))
        else:
            os.mkdir(frames_dir)
        # plotting
        ####### Begin Template Creation #######
        # n * h + (n+1) * d = 1, where n is no of cells along 1 axis,
        # h is height of one cell and d is the gap between 2 cells
        delta = 0.005
        cell_height = (1 - ((self._size + 1) * delta))/self._size
        cell_height_half = cell_height/2.0
        # plt x axis runs left to right while y runs bottom to top
        # create the full template for the board here, then just change colors
        # in the loop
        fig, axs = plt.subplots(1, 1, figsize=(8, 8), dpi=72)
        axs.axis('off')
        # add scatter points
        # axs.scatter([0, 1, 0, 1], [0, 1, 1, 0])
        ellipse_patch_list = []
        # add horizontal and vertical lines
        for i in range(self._size):
            # linewidth is dependent on axis size and hence needs
            # to be set manually
            axs.axvline((2 * i + 1)*(delta/2) + i * cell_height, 
                        color='white', lw=2)
            axs.axhline((2 * i + 1)*(delta/2) + i * cell_height, 
                        color='white', lw=2)
        for _ in range(self._size):
            ellipse_patch_list.append([0] * self._size)
        # add the large rect determining the board
        rect = Rectangle((delta, delta),
                         width=1 - 2 * delta, 
                         height=1 - 2 * delta,
                         color='forestgreen')
        axs.add_patch(rect)
        # add circle patches
        s = self._hist[0][0]
        # determine the color and alpha values
        color_array[s[:,:,0] == 1] = transition_color_list.index('black')
        color_array[s[:,:,1] == 1] = transition_color_list.index('white')
        alpha_array = (color_array != 0).astype(np.uint8)
        for i in range(self._size):
            for j in range(self._size):
                # i moves along y axis while j along x
                cell_centre = ((j + 1) * delta + (2*j + 1) * cell_height_half,\
                               (self._size - i) * delta + (2*(self._size - i) - 1) * cell_height_half)
                # a circle will be placed where a coin is
                ellipse = Ellipse(cell_centre,
                                  width=((cell_height - delta)),
                                  height=((cell_height - delta)),
                                  angle=0,
                                  color=transition_color_list[color_array[i][j]], 
                                  alpha=alpha_array[i][j])
                ellipse_patch_list[i][j] = ellipse
                # add to the figure
                axs.add_patch(ellipse_patch_list[i][j])
        # save first figure with some persistence
        fig_file_idx = 0
        for idx in range(frames_per_anim):
            if(idx == 0):
                fig.savefig('{:s}/img_{:05d}.png'.format(frames_dir, fig_file_idx), 
                                                        bbox_inches='tight')
            else:
                shutil.copyfile('{:s}/img_{:05d}.png'.format(frames_dir, 0),
                                '{:s}/img_{:05d}.png'.format(frames_dir, fig_file_idx))
            fig_file_idx += 1
        ######## End Template Creation ########
        # iterate over the game frames with animation
        for idx in tqdm(range(len(self._hist))):
            # clear figure
            # plt.cla()
            # get the board from history
            s = self._hist[idx][0]
            next_s = self._hist[idx][4]
            # prepare a single frame
            for t in range(frames_per_anim):
                # determine the color and alpha values
                # color change from black to white
                color_array[s[:,:,0] * next_s[:,:,1] == 1] = t + 1
                # color change from white to black
                color_array[s[:,:,1] * next_s[:,:,0] == 1] = frames_per_anim - t
                # no coin now and then
                color_array[s[:,:,:2].sum(2) + next_s[:,:,:2].sum(2) == 0] = 0
                # new coin placed
                color_array[(s[:,:,:2].sum(2) == 0) & (next_s[:,:,0] == 1)] = 1
                color_array[(s[:,:,:2].sum(2) == 0) & (next_s[:,:,1] == 1)] = \
                                        len(transition_color_list)-1
                # set alpha array
                alpha_array = (color_array != 0).astype(np.uint8)
                for i in range(self._size):
                    for j in range(self._size):
                        # i moves along y axis while j along x
                        # a circle will be placed where a coin is
                        ellipse_patch_list[i][j].set_color(
                                        transition_color_list[color_array[i][j]])
                        ellipse_patch_list[i][j].set_alpha(alpha_array[i][j])
                        # axs.scatter(5, 5)
                # figure is prepared, save in temp frames directory
                fig.savefig('{:s}/img_{:05d}.png'.format(frames_dir, fig_file_idx), 
                            bbox_inches='tight')
                fig_file_idx += 1
            # add some persistence before placing another new coin
            fig_file_copy_idx = fig_file_idx - 1
            for _ in range(frames_per_anim//2):
                shutil.copyfile('{:s}/img_{:05d}.png'.format(frames_dir, fig_file_copy_idx),
                '{:s}/img_{:05d}.png'.format(frames_dir, fig_file_idx))
                fig_file_idx += 1
                
        # all frames have been saved, use ffmpeg to convert to movie
        # output frame rate is different to add some persistence
        os.system('ffmpeg -y -framerate {:d} -pattern_type sequence -i "{:s}/img_%05d.png" \
          -c:v libx264 -r {:d} -pix_fmt yuv420p -vf "crop=floor(iw/2)*2:floor(ih/2)*2" {:s}'\
          .format(int(1.5 * frames_per_anim), frames_dir, int(1.5 * frames_per_anim), path))
