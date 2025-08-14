from anet import ANET
from hex import Hex
import torch
import torch.nn as nn
import random
from config import run


" Intelligent agent that uses the ANET to make moves."
class Intelligent_Agent:
    def __init__(self, path, greedy):
        self.greedy=greedy
        self.name=path
        self.model=ANET()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
    def make_move(self, game, valid_moves=None):
        if run["oht"]:
            output_anet=self.model(game)
            output_anet=nn.Softmax(dim=1)(output_anet)
            output_anet*=torch.tensor(valid_moves)
            move=self.flatten_moves_to_moves(torch.argmax(output_anet).item())
            return move
            
        else:
            output_anet=self.model(game.get_binary_board())
            output_anet=nn.Softmax(dim=1)(output_anet)
            output_anet*=torch.tensor(game.hot_encoded_valid_moves())
        
        if self.greedy==True:
            #top_moves=torch.topk(output_anet, 5)
            #print(top_moves)
            #move=random.choice(top_moves.indices[0].tolist())
            #move=game.index_flatten_moves_to_moves(move)  
            move=game.index_flatten_moves_to_moves(torch.argmax(output_anet).item())
            return move
        
        output_anet= output_anet.tolist()[0]
        move=random.choices(list(range(len(output_anet))), weights=output_anet, k=1)[0]
        move=game.index_flatten_moves_to_moves(move)
        return move

    #for OHT
    def flatten_moves_to_moves(self, index):
        return (index//7, index%7)