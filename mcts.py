"""This class holds the tree structure and relevant algorithms for
performing an iteration of monte carlo tree search"""
import numpy as np
from game_env import StateEnvBitBoard, get_total_set_bits, \
                get_set_bits_list, get_random_move_from_list
from collections import deque

class Node:
    """MCTS is a tree composed of several nodes that store statistical
    information about the simulations run so far, and are used to calculate
    the best moves

    Attributes
    ----------
    s : tuple
        tuple containing the board states as bitboards
    legal_moves : 64 bit int
        bits corresponding to legal positions are set to 1
    w : int
        the number of wins when this node was part of rollout
    n : int
        the number of simulations when this node was part of rollout
    N : int
        the number of simulations where the parent node of this node
        was part of rollout
    children : list
        list containing the child nodes for this node
    m : int
        the move played to get to this state
    terminal : int
        flag for whether this node is terminal in the tree, same as done
    parent : int
        the index of the parent of this node, in the mcts node list
    """
    def __init__(self, s, legal_moves, m=-1, terminal=0, parent=None):
        """Initializer for the node class, tree is a collection of nodes

        Parameters
        ----------
        s : tuple
            the board state represented as bitboards
        legal_moves : 64 bit int
            bits corresponding to legal positions are set to 1 in this int
        m : int (optional)
            the move played to get to this state, this denotes the position
            from the right end of array (not 64 bit)
        terminal : int (optional)
            flag denoting whether this is a leaf node
        parent : int (optional)
            index of the parent of this node in the mcts node list
        """
        self.state = s
        self.legal_moves = legal_moves
        # convert the 64 bit legal moves into a set of positions
        # for fast use later
        self.legal_moves_set = get_set_bits_list(legal_moves)
        np.random.shuffle(self.legal_moves_set)
        # to compare whether all children have been added or not
        self.total_legal_moves = get_total_set_bits(legal_moves)
        self.w = 0
        self.n = 0
        self.N = 0
        self.children = []
        # since we have shuffled the legal_moves_set, we can use total_children
        # as the idx from where we have to pick the next unexplored move
        self.total_children = 0
        self.move = m
        self.terminal = terminal
        self.parent = parent

    def add_child(self, idx):
        """Add given node as child to current node

        Parameters
        ----------
        node : idx
            index in the mcts node list
        """
        self.children.append(idx)
        self.total_children += 1

    def get_ucb1(self, c=np.sqrt(2)):
        """Get the upper confidence bound on the current node
        ucb1 = (w/n) + c*sqrt(ln(N)/n)

        Returns
        -------
        ucb1 : float
        """
        return ((0.1*self.w)/self.n) + c * np.sqrt(np.log(self.N)/self.n)


class MCTS:
    """MCTS class containing the training and inference function also
    
    Attributes
    ----------
    _c : float
        exploration coefficient
    _node_list : list
        list of nodes in the mcts tree, root is at index 0
        the children attribute will store a list of indices corresponding
        to this tree (node list), attribute parent of a node will also contain index
        referring to this tree; this was done as there are no pointers in python
        and this helps avoiding repetitions of nodes in children list etc
    _env : StateEnvBitBoard
        instance of the state environment
    """
    def __init__(self, s, legal_moves, board_size=8, c=np.sqrt(2)):
        """initializer for the MCTS class and initializes it's root

        Parameters
        ----------
        s : tuple
            the board state represented as bitboards
        legal_moves : 64 bit int
            bits corresponding to legal moves are set to 1
        c : float (optional)
            parameter for exploration in UCB, defaults to sqrt(2)
        """
        self._c = c
        self._node_list = [Node(s.copy(), legal_moves)]
        self._env = StateEnvBitBoard(board_size)

    def get_not_added_move(self, node):
        """randomly select a move from the ones not played yet

        Parameters
        ----------
        node : Node
            the node for which to select a move not added to tree yet

        Returns
        -------
        m : int
            the position (indexing from right end) where to play the move
            to use in the bitboard env, pass 1<<m instead of m
        """
        all_moves = node.legal_moves_set.copy()
        for c in node.children:
            all_moves.remove(self._node_list[c].move)
        all_moves = list(all_moves)
        return get_random_move_from_list(all_moves)

    def train(self, n=100):
        """Train the MCTS tree for n number of iterations

        Parameters
        ----------
        n : int (optional)
            the number of simulation steps to run
        """
        while(n):
            n -= 1
            ##############################
            ####### Selection Phase ######
            ##############################
            """select a node in the tree that is neither a leaf node
            nor fully explored"""
            e = 0
            while(True):
                node = self._node_list[e]
                if(node.total_legal_moves != \
                   node.total_children or \
                   node.terminal == 1):
                    # at least one unexplored move is present, stop the
                    # selection here
                    break
                else:
                    # since all nodes of previous node were explored at least
                    # once, we go to the next level and select the child 
                    # with highest ucb1
                    next_node = None
                    best_ucb1 = -np.inf
                    for idx in node.children:
                        ucb1 = self._node_list[idx].get_ucb1(self._c)
                        if(ucb1 > best_ucb1):
                            best_ucb1 = ucb1
                            next_node = idx
                    e = next_node
            # this defaults to the root in case the else condition is not run
            node, node_idx = self._node_list[e], e
            
            ##############################
            ####### Expansion Phase ######
            ##############################
            """select one of the child nodes for this node which is 
            unexplored"""
            if(not node.terminal):
                """first get a random move from the moves which have not 
                been added to the mcts tree yet"""
                # m = self.get_not_added_move(node)
                m = node.legal_moves_set[node.total_children]
                # play the game and add new node to tree (node list)
                next_state, next_legal_moves, _, done = \
                                    self._env.step(node.state, 1<<m)
                node = Node(s=next_state.copy(), legal_moves=next_legal_moves, 
                            m=m, terminal=done, parent=e)
                # add node to node list
                self._node_list.append(node)
                # add the idx in this list to the parent's children list
                self._node_list[e].add_child(len(self._node_list)-1)
                node_idx = len(self._node_list)-1

            ##############################
            ###### Simulation Phase ######
            ##############################
            """play till the end by randomly selecting moves starting from the
            newly created node (in case of terminal node this step is skipped"""
            s = node.state
            legal_moves = node.legal_moves
            if(node.terminal != 1):
                done = 0
                while(not done):
                    a = get_random_move_from_list(get_set_bits_list(legal_moves))
                    s, legal_moves, _, done = self._env.step(s, 1<<a)
            winner = self._env.get_winner(s)

            ##############################
            #### Backpropagation Phase ###
            ##############################
            """backproagate the winner value from node (from where we started
            to play) to root to update statistical parameters for each node"""
            while(True):
                node.n += 1
                # update the value of N in children
                for c in node.children:
                    self._node_list[c].N = node.n
                if(winner != -1):
                    node.w += (1-winner == self._env.get_player(node.state))
                # move one level up
                if(node.parent is None):
                    break
                else:
                    node, node_idx = self._node_list[node.parent], node.parent

    def select_move(self):
        """select the best move after the tree has been trained
        here we select the one with most number of plays

        Returns
        -------
        m : 64 bit int
            int the flag corresponding to the best move set to 1
        """
        most_plays = -np.inf
        m = -1
        for c in self._node_list[0].children:
            node = self._node_list[c]
            if(node.n > most_plays):
                m = node.move
                most_plays = node.n
        return 1 << m





