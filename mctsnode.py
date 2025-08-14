import math
import random
import torch
import torch.nn as nn
from config import training



class MCTSNode:
    def __init__(self, game, iterations):
        self.iterations = iterations
        self.game = game  # Current game 
        self.visits = 0     # Number of visits to this node
        self.score = 0      # Total score of this node
        self.children = [] # Child nodes
        self.parent = None  # Parent node
        
        if game.get_move_history():
            self.move=game.get_last_move()
        else:
            self.move=None


    def set_iterations(self, iterations):
        self.iterations = iterations
        
        
    def select(self, node):
        """
        Select a child node according to the UCB formula.
        """
        if not node.children:
            return node
        
        if node.game.current_player == 1:
            selected_child = max(node.children, key=lambda c: (c.score / (c.visits + 1)) + math.sqrt(math.log(node.visits) / (c.visits + 1)))
        else:
            selected_child = min(node.children, key=lambda c: (c.score / (c.visits + 1)) - math.sqrt(math.log(node.visits) / (c.visits + 1)))

        return self.select(selected_child)


    def expand(self, node):
        """
        Expand the children of the given node.
        """
        # Generate all possible child games and create nodes for them
        child_states = node.game.valid_moves() 
        for _ in range (len(child_states)):
            node.children.append(MCTSNode(node.game.clone(), self.iterations))
    
        for child, move in zip(node.children, child_states):
            child.game.make_move(move)
            child.move=move
            child.parent = node
            

    def simulate(self, game, anet):
        """
        Simulate a random game from the given game and return the score.
        """
        game=game.clone()
        probability_anet=training["mcts_prob_anet_simulation"]
        
        while not game.is_game_over():
            if random.random() < probability_anet:
                anet.eval()
                output_anet=nn.Softmax(dim=1)(anet(game.get_binary_board()))
                output_anet*=torch.tensor(game.hot_encoded_valid_moves())
                move=game.index_flatten_moves_to_moves(torch.argmax(output_anet).item())
            else:
                move = random.choice(game.valid_moves())
                
            game.make_move(move)
    
        # Return the score of the game
        return game.get_score()
    

    def backpropagate(self, node, score):
        """
        Update the node and its ancestors with the result of the simulation.
        """
        while node:
            node.visits += 1
            node.score += score
            node = node.parent


    def monte_carlo_tree_search(self, root, anet):
        """
        Perform Monte Carlo Tree Search for a given number of iterations.
        """
        
        for _ in range(self.iterations):
            node = root
            # Selection phase
            node = self.select(node)
             #Expansion phase
            if node.game.is_game_over():
                self.backpropagate(node, node.game.get_score())
            else:
                self.expand(node)
                # Simulation phase
                child = random.choice(node.children)
                score = self.simulate(child.game, anet)
                # Backpropagation phase
                self.backpropagate(child, score)
        

        # Return the best move based on the visit counts of the root's children
        best_child = max(root.children, key=lambda c: c.visits)
        
        
        return best_child
    

    def make_move(self, best_child, anet):
        """
        Make a move for the given game using Monte Carlo Tree Search.
        """
        root=best_child        
        move = self.monte_carlo_tree_search(root, anet)
        
        return move
    
    
    def go_to_child(self, move):
        for child in self.children:
            if child.move==move:
                return child
        return None
    
    
    #returns the state of the current board and the action probabilities
    def save_to_buffer(self):
        state = self.parent.game.get_binary_board()
        visit_count_children = [(child.visits, child.move) for child in self.parent.children] 
        action_probs = [0]*((self.parent.game.size)**2)
        for element in visit_count_children:
            action_probs[self.parent.game.move_to_index_flatten_moves(element[1])] = element[0]
        action_probs=[visit_count/sum(action_probs) for visit_count in action_probs]
        return state, action_probs

        
