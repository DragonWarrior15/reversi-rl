from flask import Flask, redirect, url_for, request, render_template, jsonify
from players import RandomPlayer, MiniMaxPlayerC, MCTSPlayerC
from game_env import StateEnvBitBoardC, get_set_bits_list, StateConverter
from random import random

app = Flask(__name__)

# a centralized list of available AI players
players = {
    'random': ['Random Player', RandomPlayer],
    'minimax': ['MiniMax Player (with alpha-beta pruning)', MiniMaxPlayerC],
    'mcts': ['Monte Carlo Tree Search (MCTS)', MCTSPlayerC]
}

# game flow related initializations
ai_player = RandomPlayer()
ai_player_coin = 0
env = StateEnvBitBoardC()
conv = StateConverter()
board_state = None

# function to handle move conversion for ai player
def ai_player_move(s, legal_moves):
    a = ai_player.move(conv.convert(s, 
                input_format='bitboard',
                output_format=ai_player.get_state_input_format()), 
                 conv.convert(legal_moves,
                    input_format='bitboard_single',
                        output_format=\
            ai_player.get_legal_moves_input_format()))
    a = conv.convert(a,
                input_format=\
                ai_player.get_move_output_format(),
                output_format='bitboard_single')
    return a

@app.route('/', methods=['GET', 'POST'])
def start_page():
    # this will just render the starting page where we can choose the
    # AI player to play against, the dict is passed for "automation"
    return render_template('index.html',
            players=dict((k, v[0]) for k,v in players.items()))

@app.route('/game_start', methods=['POST'])
def game_start():
    # this is a background service that will start the main game
    # different AI players can have different types of UI
    p = request.form['opponentPlayer']
    if p:
        # set the first player to be the one chosen by user
        ai_player = players[p][1]()
        # return the custom game UI page
        return redirect(url_for('game_ui', opp_name=players[p][0]))
    # default return
    return redirect('/')

@app.route('/game_ui/<opp_name>', methods=['GET', 'POST'])
def game_ui(opp_name=None):
    # render the starting game UI
    return render_template('game_ui.html', opp_name=opp_name)

@app.route('/game_ui/color_choice', methods=['POST'])
def game_color():
    global board_state
    # get the color in the ajax call and reset board accordingly
    c = request.form['color']
    # set ai_player color accordingly
    if(c == 'white'):
        ai_player_coin = 0
    elif(c == 'black'):
        ai_player_coin = 1
    else:
        if(random() < 0.5):
            ai_player_coin = 0
        else:
            ai_player_coin = 1
    # reset the environment
    done = 0
    board_state, legal_moves, player = env.reset()
    if(ai_player_coin == player):
        # play the move and return board
        a = ai_player_move(board_state, legal_moves)
        board_state, legal_moves, player, done = env.step(board_state, a)

    # return the boards
    return jsonify(black_board=get_set_bits_list(board_state[0]),
                   white_board=get_set_bits_list(board_state[1]),
                   legal_moves=get_set_bits_list(legal_moves),
                   player=player, done=done)

@app.route('/game_ui/step', methods=['POST'])
def game_step():
    global board_state
    # play the move chosen by human
    pos = int(request.form['position'])
    board_state, legal_moves, player, done = env.step(board_state, 1<<pos)
    # if the new player is ai, repeat the above step
    if(player == ai_player_coin):
        a = ai_player_move(board_state, legal_moves)
        board_state, legal_moves, player, done = env.step(board_state, a)
    return jsonify(black_board=get_set_bits_list(board_state[0]),
                   white_board=get_set_bits_list(board_state[1]),
                   legal_moves=get_set_bits_list(legal_moves),
                   player=player, done=done)

if __name__ == '__main__':
    app.run(debug = True)
