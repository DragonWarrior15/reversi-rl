$(document).ready(function(){
    $('#othello_grid').height($('#content').height())
    $('#othello_grid').width($('#content').height())
    var x = ""
    for(var i = 63; i >= 0; i--){
        x = x + "<div class='othello_grid_item' id='othello_grid_" + 
            i + "' onclick='on_othello_grid_click(" + i + 
            ")'><span class='othello_coin' id='othello_coin_" + i + "'></span></div>";  
    }
    $('#othello_grid').html(x);
});

function coin_choice(c){
    // c is white, black or random
    console.log(c)
    $.ajax({
        data : {color : c},
        type : 'POST',
        url : '/game_ui/color_choice'
        }).done(function(data) {
            console.log(data);
            // data has black_board, white_board, player, legal_moves
            refresh_board(data['white_board'], data['black_board'], 
                          data['legal_moves']);
        });
}

async function refresh_board(white_board, black_board, legal_moves){
    // all arguments are list of positions to update
    // reset all coin colors
    await $(".othello_coin").css("background-color", "transparent");
    if(white_board.length > 0){
        await set_color("#FFFFFF", white_board);
    }
    if(black_board.length > 0){
        await set_color("#000000", black_board);
    }
    if(legal_moves.length > 0){
        await set_color("#7FFF00", legal_moves);
    }
}

function set_color(c, l){
    // c is hexadecimal color, l is list of positions to set
    for (var i = 0; i < l.length; i++) {
        $("#othello_coin_" + l[i]).css("background-color", c);
    }
}

function on_othello_grid_click(i){
    // i is id number
    console.log($("#othello_coin_" + i).css("background-color"));
    // check if legal move
    if($("#othello_coin_" + i).css("background-color") == "rgb(127, 255, 0)"){
        // send this move to server
        $.ajax({
            data : {position : i},
            type : 'POST',
            url : '/game_ui/step'
            }).done(function(data) {
                console.log(data);
                // data has black_board, white_board, player, legal_moves
                refresh_board(data['white_board'], data['black_board'], 
                              data['legal_moves']);
            });
    }
}
