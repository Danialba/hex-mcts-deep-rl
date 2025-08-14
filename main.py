from mctsnode import MCTSNode
from randomagent import RandomAgent
from game_simulator import Game_simulator
from tic_tac_toe import Tic_Tac_Toe
from anet import ANET
from hex import Hex
from topp import Topp
import glob
import os
from intelligent_agent import Intelligent_Agent
from config import run, training, game_config, topp, custom_game


def main():
    
    if game_config["game"]=="hex":
        game=Hex(game_config["size"])
        game.set_starting_player(game_config["starting_player"])
    elif game_config["game"]=="tic_tac_toe":
        game=Tic_Tac_Toe(game_config["size"])
        game.set_starting_player(game_config["starting_player"])

    
    
    if run["train"]:
       
        player1=MCTSNode(game, training["mcts_iterations"])
        game_sim = Game_simulator(game, player1, player2=None, alternate_starting_player=False, anet=True, show_every_move=game_config["show_every_move"])
        game_sim.simulate_game(training["number_of_games"])
    
    
    elif run["topp"]:
        intelligent_agents = glob.glob(os.path.join(topp["models_path"], "*"))
        models=[]
        for i, model in enumerate(intelligent_agents):
            models.append(Intelligent_Agent(model, topp["greedy"][i]))
        if topp["add_random_agent"]:
            models.append(RandomAgent())
                
        tournament=Topp(topp["number_of_games"], game, models, alternate_starting_player=True, show_every_move=game_config["show_every_move"])
        tournament.play_tournament()
    
    
    elif run["custom_game"]:
        if custom_game["player1"]=="Human":
            player1="Human"
        elif custom_game["player1"]=="Random":
            player1=RandomAgent()
        else:
            player1=Intelligent_Agent(custom_game["player1"], greedy=custom_game["player1_greedy"])
        
        if custom_game["player2"]=="Human":
            player2="Human"
        elif custom_game["player2"]=="Random":
            player2=RandomAgent()
        else:
            player2=Intelligent_Agent(custom_game["player2"], greedy=custom_game["player2_greedy"])
        
        game_sim1=Game_simulator(game, player1, player2, custom_game["alt_starting_player"], show_every_move=game_config["show_every_move"])
        game_sim1.simulate_game(custom_game["number_of_games"])


if __name__ == "__main__":
    main()