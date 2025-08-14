from game_simulator import Game_simulator

#Class for running a tournament between any number of policies 
class Topp():

    def __init__(self, number_of_games, game, policies, alternate_starting_player, show_every_move=False):
        self.show_every_move=show_every_move
        self.alternate_starting_player=alternate_starting_player
        self.number_of_games=number_of_games
        self.game=game
        self.policies = {policy: 0 for policy in policies} #Dictionary with the policies as keys and the number of wins as values
    
    
    def play_tournament(self):
        done_playing = set()  # Store pairs of policies that have already played against each other
        for policy1 in self.policies.keys(): 
            for policy2 in self.policies.keys():
                if policy1 != policy2 and (policy1, policy2) not in done_playing and (policy2, policy1) not in done_playing:
                    self.game.initialize_game()
                    game_sim = Game_simulator(self.game, policy1, policy2, self.alternate_starting_player , show_every_move=self.show_every_move)
                    game_sim.simulate_game(self.number_of_games)
                    self.policies[policy1] += game_sim.player1_wins
                    self.policies[policy2] += game_sim.player2_wins
                    done_playing.add((policy1, policy2))  # Mark this pair as played
        
        self.print_results()
    
    def print_results(self):
        for policy in self.policies.keys():
            print(f"{policy.name} wins: {self.policies[policy]}")
            
        
                    
                    
    
                    
                    