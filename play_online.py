from intelligent_agent import Intelligent_Agent
from config import oht_config


actor = Intelligent_Agent(oht_config["player_path"], oht_config["greedy"])

from hex_client_23.ActorClient import ActorClient

class MyClient(ActorClient):

    def handle_get_action(self, state):
        binary_board= self.get_binary_board(state)
        valid_moves = self.hot_encoded_valid_moves(state)
        row, col = actor.make_move(binary_board, valid_moves)  # Your logic
        return row, col
    
    
    def get_binary_board(self, state):
        binary_board = []
        if state[0] == 1:
            binary_board.append(1)
            binary_board.append(0)
            
        elif state[0] == 2:
            binary_board.append(0)
            binary_board.append(1)
            
        for cell in state[1:]:       
            if cell == 1:
                binary_board.append(1)
                binary_board.append(0)
            elif cell == 2:
                binary_board.append(0)
                binary_board.append(1)
            else:
                binary_board.append(0)
                binary_board.append(0)
                    
        return binary_board
    
    def hot_encoded_valid_moves(self, state):
        valid_moves = []
        for i in range (len(state)-1):
            if state[i+1]==0:
                valid_moves.append(1)
            else:
                valid_moves.append(0)
                
        return valid_moves
        
    
    # Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient()
    client.run(mode='league')


