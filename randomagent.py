import random
class RandomAgent:
    
    def __init__(self):
        self.name="RandomAgent"
        self=self
    
    #Makes random move
    def make_move(self, game):
        return random.choice(game.valid_moves())
    
    
    