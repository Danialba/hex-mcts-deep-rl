
# There are 4 different modes to run the program: train, topp, custom_game, oht.
run= {
    "train": True,
    "topp": False,
    "custom_game": False,
    "oht":False
}

game_config = {
    "game": "hex", # Game to play, could be "hex" or "tic_tac_toe"
    "size": 4, # Size of the board
    "starting_player": 1, # Starting player, 1 or -1. Corresponding to player X and player O in hex.
    "show_every_move": True, # Show the board after every move in the real game
}

training = {
    "number_of_games": 100, # Number of games to play
    "mcts_iterations": 500, # Number of MCTS iterations
    "batch_size": 128, # Number of samples to take from the replay buffer
    "mcts_prob_anet_simulation": 0.3, # Probability of using the ANET during simulation for each move
    "cache": True, # Cache the models during training
    "M":5, #number of models to cache during training in addition to the default model. Only used if cache is True. 
    "save_to_folder": "saved_models/", # Folder to save the models. Only used if cache is True
    "plot_loss": True # Plot the loss after training
}

anet_config={
    "input_size": 2*(game_config["size"]**2)+2, # Input size for the ANET
    "hidden_layers": (128,256,512,256,128), # Number of neurons in each hidden layer. for pre- saved models for demo: (128,256,512,256,128)
    "activation": ("relu", "relu", "relu", "relu", "relu", "relu"), # Activation functions for each hidden layer, could be "relu", "tanh", "sigmoid", "linear"
    "output_size": game_config["size"]**2, # Output size for the ANET
    "lr": 0.001, # Learning rate for the optimizer
    "optimizer": "Adagrad" # Optimizer to use, could be "Adam", "SGD", "RMSprop", "Adagrad"
}


topp = {
    "number_of_games": 100, # Number of games to play in the tournament between each model
    "models_path": "saved_models", # Path to the models. Each model in this folder will be played against each other
    "add_random_agent": False, # Add a random agent to the tournament
    "greedy": [False]*6, # must be of same length as number of models. If True, the agent will always choose the move with the highest probability. If false it will choose a move randomly weighted on the probability distribution
}


custom_game = {
    "player1": "Random",
    "player1_greedy": False,
    "player2": "Model_5x5_for_demo/model_200_anet.pth",
    "player2_greedy": True,
    "number_of_games": 1,
    "alt_starting_player": True
}

oht_config={
    "player_path": "Model_7x7/model_275_anet.pth",
    "greedy": True,
}
