// run this command to build the shared library
// gcc -fPIC -shared -o cfns.dll cfns.c
// #include <stdio.h>
void get_next_board(unsigned long long s0, unsigned long long s1, 
                    int p, unsigned long long a,
                    unsigned long long* ns0, unsigned long long* ns1){

    /*Determine the updated bitboard after performing action a
    using player passed to the function

    Parameters
    ----------
    s : list
        contains the bitboards for white, black coins and 
        the current player
    a : int (64 bit)
        the bit where action needs to be done is set to 1

    Returns
    -------
    s_next : list
        updated bitboards
    */
    unsigned long long board_p = s0, board_notp = s1;
    if(p == 1){
        board_p = s1; board_notp = s0;
    }
    // keep a global updates master
    unsigned long long update_master = 0, m, c;
    // left
    m = 0;
    c = board_notp & (a << 1) & 18374403900871474942llu;
    while(c & board_notp){
        m = m | c;
        c = (c << 1) & 18374403900871474942llu;        
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // right
    m = 0;
    c = board_notp & (a >> 1) & 9187201950435737471llu;
    while(c & board_notp){
        m = m | c;
        c = (c >> 1) & 9187201950435737471llu;        
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // top
    m = 0;
    c = board_notp & (a << 8) & 18446744073709551360llu;
    while(c & board_notp){
        m = m | c;
        c = (c << 8) & 18446744073709551360llu;        
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // bottom
    m = 0;
    c = board_notp & (a >> 8) & 72057594037927935llu;
    while(c & board_notp){
        m = m | c;
        c = (c >> 8) & 72057594037927935llu;        
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // right_top
    m = 0;
    c = board_notp & (a << 7) & 9187201950435737344llu;
    while(c & board_notp){
        m = m | c;
        c = (c << 7) & 9187201950435737344llu;
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // left_top
    m = 0;
    c = board_notp & (a << 9) & 18374403900871474688llu;
    while(c & board_notp){
        m = m | c;
        c = (c << 9) & 18374403900871474688llu;
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // right_bottom
    m = 0;
    c = board_notp & (a >> 9) & 35887507618889599llu;
    while(c & board_notp){
        m = m | c;
        c = (c >> 9) & 35887507618889599llu;
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    // left_bottom
    m = 0;
    c = board_notp & (a >> 7) & 71775015237779198llu;
    while(c & board_notp){
        m = m | c;
        c = (c >> 7) & 71775015237779198llu;
    }
    if(!(c & board_p)){
        m = 0;
    }
    else{
        update_master = update_master | m;
    }
    
    // all directions searched, now update the bitboards
    board_p = board_p | update_master | a;
    board_notp = board_notp - update_master;
    // return
    if(p == 0){
        *ns0 = board_p; *ns1 = board_notp;
    }
    else{
        *ns0 = board_notp; *ns1 = board_p;
    }
}


void legal_moves_helper(unsigned long long s0, unsigned long long s1,
                          int p, unsigned long long* moves){
    /*Get the bitboard for legal moves of given player

    Parameters
    ----------
    s : list
        contains the bitboard for black coins, white coins and
        the int representing player to play

    Returns
    m : int (64 bit)
        bitboard representing the legal moves for player p
    */
    unsigned long long board_p = s0, board_notp = s1;
    if(p == 1){
        board_p = s1; board_notp = s0;
    }
    // keep a global updates master
    unsigned long long c, e, m = 0;
    // define the empty set
    e = ~(board_p | board_notp);
    // for every direction, run the while loop to get legal moves
    // the while loop traverses paths of same coloured coins
    // get the set of positions where there is a coin of opposite player
    // to the direction of player to play
    // left
    c = board_notp & (board_p << 1) & 18374403900871474942llu;
    while(c){
        // if immediately to direction is empty, this is a legal move
        m = m | (e & (c << 1) & 18374403900871474942llu);
        // we can continue the loop till we keep encountering opposite player
        c = board_notp & (c << 1) & 18374403900871474942llu;
    }
    // right
    c = board_notp & (board_p >> 1) & 9187201950435737471llu;
    while(c){
        // if immediately to direction is empty, this is a legal move
        m = m | (e & (c >> 1) & 9187201950435737471llu);
        // we can continue the loop till we keep encountering opposite player
        c = board_notp & (c >> 1) & 9187201950435737471llu;
    }
    // top
    c = board_notp & (board_p << 8) & 18446744073709551360llu;
    while(c){
        // if immediately to direction is empty, this is a legal move
        m = m | (e & (c << 8) & 18446744073709551360llu);
        // we can continue the loop till we keep encountering opposite player
        c = board_notp & (c << 8) & 18446744073709551360llu;        
    }
    // bottom
    c = board_notp & (board_p >> 8) & 72057594037927935llu;
    while(c){
        // if immediately to direction is empty, this is a legal move
        m = m | (e & (c >> 8) & 72057594037927935llu);
        // we can continue the loop till we keep encountering opposite player
        c = board_notp & (c >> 8) & 72057594037927935llu;
    }
    // right_top
    c = board_notp & (board_p << 7) & 9187201950435737344llu;
    while(c){
        // if immediately to direction is empty, this is a legal move
        m = m | (e & (c << 7) & 9187201950435737344llu);
        // we can continue the loop till we keep encountering opposite player
        c = board_notp & (c << 7) & 9187201950435737344llu;
    }
    // left_top
    c = board_notp & (board_p << 9) & 18374403900871474688llu;
    while(c){
        // if immediately to direction is empty, this is a legal move
        m = m | (e & (c << 9) & 18374403900871474688llu);
        // we can continue the loop till we keep encountering opposite player
        c = board_notp & (c << 9) & 18374403900871474688llu;
    }
    // right_bottom
    c = board_notp & (board_p >> 9) & 35887507618889599llu;
    while(c){
        // if immediately to direction is empty, this is a legal move
        m = m | (e & (c >> 9) & 35887507618889599llu);
        // we can continue the loop till we keep encountering opposite player
        c = board_notp & (c >> 9) & 35887507618889599llu;
    }
    // left_bottom
    c = board_notp & (board_p >> 7) & 71775015237779198llu;
    while(c){
        // if immediately to direction is empty, this is a legal move
        m = m | (e & (c >> 7) & 71775015237779198llu);
        // we can continue the loop till we keep encountering opposite player
        c = board_notp & (c >> 7) & 71775015237779198llu;
    }
    // return final legal moves
    // return m;
    *moves = m;
}