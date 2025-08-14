
from mctsnode import MCTSNode
from rbuf import RBUF
from anet import ANET
import matplotlib.pyplot as plt
import ast
import torch
from config import training

class Game_simulator:
    
    def __init__(self, game, player1, player2=None, alternate_starting_player=False, anet=None, show_every_move=False):
        self.show_every_move=show_every_move
        self.anet=anet
        self.game = game
        self.player1=player1
        self.player2=player2
        self.alternate_starting_player= alternate_starting_player
        self.player1_wins=0
        self.player2_wins=0
        self.ties=0
        
    
    def make_move(self, player, current_node=None):
        if self.anet:
            best_child_node = player.make_move(current_node, self.anet) #returns the best child based in the current node
            self.game.make_move(best_child_node.game.get_last_move()) #makes the move in the game
            if self.show_every_move:
                self.game.print_current_board()
                print("\n")
            return best_child_node
        
        if player == "Human":
            self.game.make_move(ast.literal_eval(input("Enter the number of the square you want to mark : ")))
        
        else:
            move=player.make_move(self.game)
            self.game.make_move(move)
            
        if self.show_every_move:
            self.game.print_current_board()
            print("\n")
            
        
    def run_two_players(self, number_of_games):
        starting_player=1
        players=[self.player1, self.player2]

        for game in range (number_of_games):
            if self.alternate_starting_player and game != 0:
                players.reverse()
                starting_player*=-1
            
            self.game.initialize_game()
            self.game.set_starting_player(starting_player)
                
            if self.show_every_move:
                self.game.print_current_board()
                print("\n")
            
            #play game
            while not self.game.is_game_over():
                self.make_move(players[0])
                if not self.game.is_game_over():
                    self.make_move(players[1])
            
            if self.game.current_winner==1:
                self.player1_wins+=1
            elif self.game.current_winner==-1:
                self.player2_wins+=1
            else:
                self.ties+=1   
                
        print(f"Player 1 wins: {self.player1_wins}")
        print(f"Player 2 wins: {self.player2_wins}")
        print(f"Ties: {self.ties}") 
                
    
    
    def run_training(self, number_of_games):
        starting_player=1
        rbuf=RBUF() # init replay buffer
        self.anet=ANET() # init neural network
        
        for game in range (number_of_games):
            self.game.initialize_game()
            self.game.set_starting_player(starting_player)
            self.player1=MCTSNode(self.game.clone(), self.player1.iterations)    

            print(f"Game number {game+1}")
            current_node = self.player1   
            
            #play game         
            while not self.game.is_game_over():
                current_node = self.make_move(self.player1, current_node)
                rbuf.add(current_node.save_to_buffer())
                
            if self.game.current_winner==1:
                self.player1_wins+=1
            elif self.game.current_winner==-1:
                self.player2_wins+=1
            else:
                self.ties+=1       
                
                 
            training_data=rbuf.sample(training["batch_size"])
            if training_data:
                self.anet.train_model(training_data)
            
            if training["cache"]:
                if (game+1)%(number_of_games/training["M"]) == 0 or game==0:
                    torch.save(self.anet.state_dict(), f"{training["save_to_folder"]}model_{game+1}_anet.pth")
  
            
                
        print(f"Player 1 wins: {self.player1_wins}")
        print(f"Player 2 wins: {self.player2_wins}")
        print(f"Ties: {self.ties}")
        
        if training["plot_loss"]:
            self.plot_loss_history(self.anet.loss_history)  


    def plot_loss_history(self, loss_history):
        plt.plot(loss_history)
        plt.xlabel('Training Iterations')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

                
    def simulate_game(self, number_of_games):
        if self.anet:
            self.run_training(number_of_games)
        else:
            self.run_two_players(number_of_games)            
                
                

