# python game_server.py
from flask import Flask, request, render_template, jsonify
from players import RandomPlayer, MiniMaxPlayerC, MCTSPlayerC
from game_env import StateEnvBitBoardC, get_set_bits_list, StateConverter
from random import random

app = Flask(__name__)

"""a centralized list of available AI players
this is used to display the first page where user can select
which ai to play against"""
ai_players = {
    'random': ['Random Player', RandomPlayer],
    'minimax': ['MiniMax Player (with alpha-beta pruning)', MiniMaxPlayerC],
    'mcts': ['Monte Carlo Tree Search (MCTS)', MCTSPlayerC]
}

# game flow related initializations, global variables
ai_player = None
ai_player_coin = 0
env = StateEnvBitBoardC()
conv = StateConverter()
board_state = None
board_legal_moves = None

def ai_player_move(s, legal_moves):
    """
    Same function is used in the Game class from game_env
    this function handles all move conversion related manioulations

    Parameters
    ----------
    s : list
        [black bitboard, white bitboard, player]
    legal_moves : 64 bit int
        the bitboard for legal moves

    Returns
    -------
    a : 64 bit int
        single bit corresponding to move position is set
    """
    # get the move from ai
    global conv, ai_player
    a = ai_player.move(conv.convert(s, 
                input_format='bitboard',
                output_format=ai_player.get_state_input_format()), 
                 conv.convert(legal_moves,
                    input_format='bitboard_single',
                        output_format=\
            ai_player.get_legal_moves_input_format()))
    # convert the move to bitboard
    a = conv.convert(a,
                input_format=\
                ai_player.get_move_output_format(),
                output_format='bitboard_single')
    return a

@app.route('/', methods=['GET', 'POST'])
def start_page():
    """
    this will just render the starting page where we can choose the
    AI player to play against, the dict is passed for "automation"

    Returns
    -------
    game_ui : html
        the html page corresponding to home
    """
    return render_template('game_ui.html',
            players=dict((k, v[0]) for k,v in ai_players.items()))

@app.route('/ai_choice', methods=['POST'])
def ai_choice():
    """
    this function is called after selecting the ai to play against
    already initialized ai is selected here and the html corresponding
    to coin selection is sent from here

    Returns
    -------
    json : json
        the name of ai to display, the new html for #right_panel
    """
    global ai_player
    c = request.form['ai_player']
    d = request.form['difficulty']
    # convert d to percentage with respect to 10
    d = (int(d)-1)/(10-1.0)
    ai_player = ai_players[c][1]
    # initialize with difficulty
    if(c == 'minimax'):
        ai_player = ai_player(depth=int(d*(9-1) + 1))
    elif(c == 'mcts'):
        ai_player = ai_player(n_sim=int(d*(50000-1) + 100))
    else: # random
        ai_player = ai_player()
    # read the html
    with open('templates/coin_choice_btn.html', 'r') as f:
        coin_choice_html = f.read()
    # append html for reset button
    with open('templates/reset.html', 'r') as f:
        coin_choice_html += f.read()
    # return in json format
    return jsonify(ai_player_name=ai_players[c][0], new_html=coin_choice_html)

@app.route('/coin_choice', methods=['POST'])
def coin_choice():
    """
    this function is called after the player has chosen which coin
    to play with, the environment is reset here and the board
    data is returned in json format

    Returns
    -------
    json : json
        black_board, white_board, legal_moves, player, 
        done (if game ended or not), ai_player_coin (0/1), score_display_html
    """
    global board_state, board_legal_moves, ai_player_coin, env
    # get the color in the ajax call and reset board accordingly
    c = request.form['color']
    # set ai_player color accordingly
    if(c == 'white'):
        ai_player_coin = 0
    elif(c == 'black'):
        ai_player_coin = 1
    else: # c == 'random'
        if(random() < 0.5):
            ai_player_coin = 0
        else:
            ai_player_coin = 1
    # reset the environment
    done = 0
    board_state, board_legal_moves, player = env.reset()
    # read the html to render for score display
    with open('templates/score_display.html', 'r') as f:
        score_display_html = f.read()
    # append the reset button to html
    with open('templates/reset.html', 'r') as f:
        score_display_html += f.read()
    # modify this html if necessary
    if(ai_player_coin == 1):
        score_display_html = score_display_html\
                            .replace('AI (Black)', 'AI (White)')\
                            .replace('You (White)', 'You (Black)')
    # return the boards and other data, html
    return jsonify(black_board=get_set_bits_list(board_state[0]),
                   white_board=get_set_bits_list(board_state[1]),
                   legal_moves=get_set_bits_list(board_legal_moves),
                   player=player, done=done, ai_player_coin=ai_player_coin,
                   score_display_html=score_display_html)

@app.route('/step', methods=['POST'])
def game_step():
    """
    this function is called everytime a player/ai wants to play a move
    on the board, the position passed is in [0,63] where 0 is bottom right
    corner

    Returns
    -------
    json : json
        black_board, white_board, legal_moves, player, done
    """
    global board_state, board_legal_moves
    # play the move chosen by human
    pos = int(request.form['position'])
    # ai steps if pos is -1
    if(pos == -1):
        a = ai_player_move(board_state, board_legal_moves)
    else: # human player move
        a = 1<<pos
    board_state, board_legal_moves, player, done = env.step(board_state, a)
    # return new states
    return jsonify(black_board=get_set_bits_list(board_state[0]),
                   white_board=get_set_bits_list(board_state[1]),
                   legal_moves=get_set_bits_list(board_legal_moves),
                   player=player, done=done)

if __name__ == '__main__':
    app.run(debug = True)
