// run this command to build the shared library
// gcc -fPIC -shared -o mcts.dll mcts.c
#include "game_env_fn.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
struct Node{
    // define all the variables to store here
    // refer to mcts.py for the complete documentation
    unsigned long long s0;
    unsigned long long s1;
    unsigned int p;
    // legal moves
    unsigned long long legal_moves;
    unsigned int *legal_moves_set;
    // total legal moves
    unsigned int total_legal_moves;
    // two asteriks to create an array of pointers
    // https://stackoverflow.com/questions/15397728/c-pointer-to-array-of-pointers-to-structures-allocation-deallocation-issues
    struct Node **children;
    unsigned int w;
    unsigned int n;
    unsigned int N;
    // since we have shuffled the legal_moves_set, we can use total_children
    // as the idx from where we have to pick the next unexplored move
    unsigned int total_children;
    unsigned int m; // move position, not 64 bit one
    unsigned int terminal; // whether leaf node
    struct Node *parent; // pointer to parent node
};

void node_init(struct Node *node, unsigned long long s0, unsigned long long s1,
               unsigned int p, unsigned long long legal_moves,
               unsigned int move, unsigned int terminal,
               struct Node *parent){
    node->s0 = s0; node->s1 = s1; node->p = p; node->legal_moves = legal_moves;
    // calculate the total set bits and prepare the moves list array
    node->total_legal_moves = get_total_set_bits(legal_moves);
    // int arr[node->total_legal_moves]; doesnt work, use malloc for dynamic
    node->legal_moves_set = malloc(node->total_legal_moves * sizeof(int));
    get_set_bits_array(legal_moves, node->legal_moves_set);
    // for(int i = 0; i < node->total_legal_moves;i++){
        // printf("%d %d %d\n", 100, i, node->legal_moves_set[i]);
    // }
    // set the parameters to 0 related to ucb
    node->w = 0; node->n = 0; node->N = 0;
    // children array
    node->children = malloc(node->total_legal_moves * sizeof(struct Node*));
    node->total_children = 0;
    // move
    node->m = move;
    // terminal or leaf node
    node->terminal = terminal;
    // parent
    node->parent = parent;
}

double get_ucb1(struct Node *node){
    double w = node->w, n = node->n, N = node->N;
    double ans = (w/n) + (sqrt(2) * sqrt(log(N)/n));
    return ans;
}


unsigned int move(unsigned long long s0, unsigned long long s1, unsigned int p,
                      unsigned long long legal_moves, unsigned int n_sim){
    printf("%d\n", n_sim);
    /*s0-> black bitboard, s1-> white bitboard, p-> player,
    legal_moves-> legal moves for current state and player,
    n-> total simulations*/
    // prepare the root node of the tree
    struct Node root;
    node_init(&root, s0, s1, p, legal_moves, 0, 0, 0);
    // define some variables to prevent redeclarations
    unsigned long long ns0, ns1, nlegal_moves;
    unsigned int np, done, total_set_bits, most_plays = n_sim+1, idx;
    int winner, m;
    // start the simulations
    struct Node *node;
    while(n_sim--){
        printf("%d\n", n_sim);
        /*############################
        ####### Selection Phase ######
        ############################*/
        /*select a node in the tree that is neither a leaf node
        nor fully explored*/
        node = &root;
        while(1){
            printf("%d\n", 101);
            printf("%d %d\n", 101, node);
            printf("%d %d %d %llu\n", 101, node->total_legal_moves, node->total_children, node->s0);
            printf("%d %d\n", 101, node->total_children);
            for(int i = 0; i < node->total_children; i++){
                printf("%d %d %d %d\n", 101, i, node->legal_moves_set[i], node->children[i]);
            }
            if(node->total_legal_moves != node->total_children || 
               node->terminal == 1){
                // at least one unexplored move is present, stop the
                // selection here
                break;
            }else{
                // since all nodes of previous node were explored at least
                // once, we go to the next level and select the child 
                // with highest ucb1
                double best_ucb1 = 0, ucb1; idx = 0;
                for(int i = 0; i < node->total_children; i++){
                    printf("%d %d %llu %d\n", node->children[i]->total_legal_moves, 
                           node->children[i]->total_children, node->children[i]->s0,
                           node->children[i]->m);
                    ucb1 = get_ucb1(node);
                    if(ucb1 > best_ucb1){
                        best_ucb1 = ucb1;
                        idx = i;
                    } 
                }
                node = node->children[idx];
                printf("%d, %d\n", 1012, node);
            }
        }

        /*############################
        ####### Expansion Phase ######
        ############################*/
        /*select one of the child nodes for this node which is unexplored*/
        // printf("%d %d\n", 102, node->terminal);
        if(!node->terminal){
            // printf("%d\n", 102);
            /*first get a random move from the moves which have not 
            been added to the mcts tree yet*/
            m = node->legal_moves_set[node->total_children];
            // printf("%d %d %d\n", 102, m, node->legal_moves_set[node->total_children]);

            // play the game and add new node to tree
            step(node->s0, node->s1, node->p, 1<<m, &ns0, &ns1, 
                  &nlegal_moves, &np, &done);
            // create the new node
            // struct Node next_node does not work as it allocates 
            // the same address
            struct Node *next_node = malloc(sizeof(struct Node));
            node_init(next_node, ns0, ns1, np, nlegal_moves, m, done, node);
            // add the idx in this list to the parent's children list
            // also update the related values
            node->children[node->total_children] = next_node;
            // printf("%d %d %d %llu %d %llu %d\n", 102, node->children[node->total_children], 
                       // next_node, next_node->s0, next_node->m, node->children[node->total_children]->s0, 
                       // node->children[node->total_children]->m);
            node->total_children++;
            // change the node to next_node
            node = next_node;
        }

        /*############################
        ###### Simulation Phase ######
        ############################*/
        /*play till the end by randomly selecting moves starting from the
        newly created node (in case of terminal node this step is skipped)*/
        if(!node->terminal){
            done = 0; s0 = node->s0; s1 = node->s1; p = node->p;
            legal_moves = node->legal_moves;
            while(!done){
                // printf("%d\n", 1031);
                total_set_bits = get_total_set_bits(legal_moves);
                // printf("%d\n", 1032);
                // reinitialize again and again as it's size is variable
                int move_list[total_set_bits];
                get_set_bits_array(legal_moves, move_list);
                // pick a random move, modulo ensures max is not out of array
                m = move_list[rand()%total_set_bits];
                step(s0, s1, p, 1<<m, &ns0, &ns1, &nlegal_moves, &np, &done);
                s0 = ns0; s1 = ns1; p = np; legal_moves = nlegal_moves;
            }
            winner = get_winner(s0, s1);
        }

        /*############################
        #### Backpropagation Phase ###
        ############################*/
        /*backproagate the winner value from node (from where we started
        to play) to root to update statistical parameters for each node*/
        while(1){
            node->n++;
            // update the value of N in children
            for(int i = 0; i < node->total_children; i++){
                node->children[i]->N = node->n;
            }
            if(winner == node->p){
                node->w++;
            }
            // move one level up
            if(!node->parent){break;}
            else{
                node = node->parent;
            }
        }
    }

    // free memory
    // free(node);

    /*select the best move after the tree has been trained
    here we select the one with most number of plays*/
    // most_plays already initialized above
    idx = 0;
    for(int i = 0; i < root.total_children; i++){
        if(root.children[i]->n > most_plays){
            m = root.children[i]->m;
            most_plays = root.children[i]->n;
        }
    }
    // convert to 64 bit in python as data type can be modified
    // when passing from C to python
    return m;
}
