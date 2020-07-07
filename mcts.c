// run this command to build the shared library
// gcc -fPIC -shared -o mcts.dll mcts.c
#include "game_env_fn.c"
int fun2(int a){
    return fun1(a);
}