import copy

class Tic_Tac_Toe:
    
    def __init__(self, size):
        self.size=size
        self.move_history = []
        self.board = [0]*size**2
        self.current_player = None
        self.current_winner = None

        
    def initialize_game(self):
        self.board = [0]*9
        self.move_history=[]
        self.current_player=None
        self.current_winner=None

    def set_starting_player(self, starting_player):
        self.current_player=starting_player
    
    def print_board(self):
        print(self.board[:3])
        print(self.board[3:6])
        print(self.board[6:9])
            
    def make_move(self, square):
        if self.board[square] == 0:
            self.board[square] = self.current_player
            self.move_history.append(square)
            self.current_player *= -1
            if len(self.move_history) >= 5:
                self.check_winner(self.board)
                
    def get_move_history(self):
        return self.move_history

    
    def valid_moves(self):
        return [move for move in range(9) if move not in self.move_history]
    
    def check_winner(self, board):
        # Define winning combinations
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        # Check for a winner
        for combo in winning_combinations:
            if board[combo[0]] == board[combo[1]] == board[combo[2]] != 0:
                self.current_winner= board[combo[0]] 
                return True
        return False
    
    def is_board_full(self):
        return len(self.move_history) == self.size**2
    
    def get_score(self):
        if self.current_winner == 1:
            return 1
        elif self.current_winner == -1:
            return -1
        else:
            return 0
    
    def get_last_move(self):
        return self.move_history[-1]
    
    def is_game_over(self):
        return self.is_board_full() or self.current_winner is not None
    
    def get_binary_board(self):
        binary_board = []
        if self.current_player == 1:
            binary_board.append(1)
            binary_board.append(0)
        else:
            binary_board.append(0)
            binary_board.append(1)
            
        for square in self.board:
            if square == 1:
                binary_board.append(1)
                binary_board.append(0)
            elif square == -1:
                binary_board.append(0)
                binary_board.append(1)
            else:
                binary_board.append(0)
                binary_board.append(0)
                
        return binary_board
    
    def index_flatten_moves_to_moves(self, index):
        return index
        
    
            
    def clone(self):
        return copy.deepcopy(self)

    def hot_encoded_valid_moves(self):
        hot_encoded_moves = [0]*9
        for move in self.valid_moves():
            hot_encoded_moves[move] = 1
        return hot_encoded_moves
    