// global variables to track game
var ai_player_coin = 0;
var current_player_coin = 1;
var legal_moves_color = "rgb(127, 255, 0)";

$(document).ready(function(){
    // this part renders the complete grid of othello game
    // get the height of content div and set othello grid as same 
    $('#othello_grid').height($('#content').height())
    $('#othello_grid').width($('#content').height())
    // prepare the grid with coins
    var x = ""
    for(var i = 63; i >= 0; i--){
        x = x + "<div class='othello_grid_item' id='othello_grid_" + 
            i + "' onclick='on_othello_grid_click(" + i + 
            ")'><span class='othello_coin' id='othello_coin_" + i + "'></span></div>";  
    }
    // set the html
    $('#othello_grid').html(x);
});

async function ai_choice(c){
    /*c is the ai player to play against, refer to game_server.py
    for correct definition, async since we will wait till we get
    data from the ajax call*/
    // send the select ai name to the server
    data = await $.ajax({
                data : {ai_player : c},
                type : 'POST',
                url : '/ai_choice'
            });
    // data has player name
    $("#page_title").html("Othello Game against " + data['ai_player_name']);
    // display the coin selection button
    $("#right_panel").html(data['new_html']);
}

async function coin_choice(c){
    /*c is white, black or random, the coin choice, async function since
    we will wait till the ajax call is complete*/
    // send the selected color to the server
    data = await $.ajax({
        data : {color : c},
        type : 'POST',
        url : '/coin_choice'
    });
    // set the ai player coin
    ai_player_coin = data['ai_player_coin'];
    // get the color that has to play
    current_player_coin = data['player'];
    // call the board refresh logic
    // data has black_board, white_board, player, legal_moves, done
    refresh_board(data['white_board'], data['black_board'], 
                      data['legal_moves'], data['done'], data['player']);
    // remove the coin choice html, show score
    $("#right_panel").html(data['score_display_html']);
    // display whose turn it is to play, step the ai if its turn is first
    if(current_player_coin == ai_player_coin){
        $("#score_display_turn").html("AI's turn");
        ai_step();
    }
}

async function score_set(white_board, black_board){
    /*updates the scores after board refresh using length of lists

    Parameters
    ----------
    white board : list
        list of indices to set as white
    black_board : list
        list of indices to set as white
    */
    if(ai_player_coin == 0){
        $("#score_ai").html(black_board.length);
        $("#score_you").html(white_board.length);
    }else{
        $("#score_ai").html(white_board.length);
        $("#score_you").html(black_board.length);        
    }
}

async function ai_step(){
    /*step function for ai, separate function is used since it is
    possible that the AI has to run multiple times, and we want to
    give a small gap between each move, position is -1 so that the
    server does not confuse with a human move*/
    // wait before executing
    await sleep(500);
    // wait for the ajax call to complete
    data = await $.ajax({data : {position : -1},
                type : 'POST',
                url : '/step'
            });
    // update the next player
    current_player_coin = data['player'];
    // refresh coins
    refresh_board(data['white_board'], data['black_board'], 
                  data['legal_moves'], data['done'], data['player']);
    // in case the next move is also of ai, run this function again
    if(data['player'] == ai_player_coin){
        ai_step();
    } 
}

async function refresh_board(white_board, black_board, legal_moves, 
                             done, player){
    /*this function will change the colors of all coins according to
    the received list of white coins, black coins, legal_moves

    Parameters
    ----------
    white_board : list
        indices where white coins to be placed
    black_board : list
        indices where black coins to be placed
    legal_moves : list
        indices where legal moves must be represented
    done : int
        0/1 whether the game has ended
    player : int
        0/1 denoting the color of next coin to be played
    */
    // reset all coin colors, dimensions
    await $(".othello_coin").css({"background-color": "transparent"});
    // set colors based on board/legal moves
    if(white_board.length > 0){
        await set_color("#F2F2F2", white_board);
    }
    if(black_board.length > 0){
        await set_color("#000000", black_board);
    }
    if(legal_moves.length > 0){
        await set_color(legal_moves_color, legal_moves);
    }
    // set the scores
    await score_set(white_board, black_board);
    // display who to play next
    if(ai_player_coin == player){
        $("#score_display_turn").html("AI's turn");
    }else{
        $("#score_display_turn").html("Your turn");
    }
    // display winner if game ends
    if(done == 1){
        $("#score_display_turn").html("Game Over");
        $("#winner").css('display', 'block');
        if(white_board.length == black_board.length){
            $("#winner").html("Tie");
        }else if(white_board.length > black_board.length){
            if(ai_player_coin == 1){
                $("#winner").html("Winner is AI");
            }else{
                $("#winner").html("Winner is You");
            }
        }else{
            if(ai_player_coin == 0){
                $("#winner").html("Winner is AI");
            }else{
                $("#winner").html("Winner is You");
            }            
        }
    }
}

function set_color(c, l){
    /*set color c to list of indices l

    Parameters
    ----------
    c : color
        white, black or green
    l : list
        list of indices to update color of othello coins
    */
    for (var i = 0; i < l.length; i++) {
        $("#othello_coin_" + l[i]).css("background-color", c);
    }
}

async function on_othello_grid_click(i){
    /*event handler that is fired whenever any "cell" of the othello
    grid is clicked

    Parameters
    ----------
    i : int
        the index of cell clicked
    */
    // i is id number
    // check if legal move and click happens when human's turn
    if($("#othello_coin_" + i).css("background-color") == legal_moves_color &&
       ai_player_coin != current_player_coin){
        // send this move to server
        data = await $.ajax({
                    data : {position : i},
                    type : 'POST',
                    url : '/step'
            });
        // update the next player to play
        current_player_coin = data['player'];
        // refresh the board
        refresh_board(data['white_board'], data['black_board'], 
                      data['legal_moves'], data['done'], data['player']);
        // play the ai move if ai's turn
        if(current_player_coin == ai_player_coin){
            ai_step();
        }
    }
}

function sleep(ms) {
    /*to wait before executing next part*/
    return new Promise(resolve => setTimeout(resolve, ms));
}