from disjoint_set import DisjointSet

class Hex:
    
    def __init__(self, size):
        
        "Code based on the disjoint-set architecture for check-wins from:  https://gist.github.com/inside-code-yt/6ffdac353a448440d8f6d649d4ed3e22"
        
        self.size=size
        self.board = [[0]*self.size for _ in range(self.size)]
        self.cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.top_node = (-1, 0)
        self.bottom_node = (self.size, 0)
        self.left_node = (0, -1)
        self.right_node = (0, self.size)
        self.ds_player_1 = DisjointSet(self.cells + [self.top_node, self.bottom_node])
        self.ds_player_2 = DisjointSet(self.cells + [self.left_node, self.right_node])
        self.move_history = []
        self.current_player = None
        self.current_winner = None
        for i in range(self.size):
            self.ds_player_1.union((0, i), self.top_node)
            self.ds_player_1.union((self.size-1, i), self.bottom_node)
            self.ds_player_2.union((i, 0), self.left_node)
            self.ds_player_2.union((i, self.size-1), self.right_node)

    def get_move_history(self):
        return self.move_history
    
    def initialize_game(self):
        self.board = [[0]*self.size for _ in range(self.size)]
        self.move_history = []
        self.current_player = None
        self.current_winner = None
        self.cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        self.ds_player_1 = DisjointSet(self.cells + [self.top_node, self.bottom_node])
        self.ds_player_2 = DisjointSet(self.cells + [self.left_node, self.right_node])
        for i in range(self.size):
            self.ds_player_1.union((0, i), self.top_node)
            self.ds_player_1.union((self.size-1, i), self.bottom_node)
            self.ds_player_2.union((i, 0), self.left_node)
            self.ds_player_2.union((i, self.size-1), self.right_node)

        
        
    def set_starting_player(self, starting_player):
        self.current_player = starting_player
        
    def valid_moves(self):
        return [(i, j) for i in range(self.size) for j in range(self.size) if self.board[i][j] == 0]
    
      
    def make_move(self, move):
        x = move[0]
        y = move[1]
        if self.board[x][y] == 0:
            self.board[x][y] = self.current_player
            for nei_x, nei_y in [(x+1, y), (x+1, y-1), (x, y+1), (x, y-1), (x-1, y), (x-1, y+1)]:
                if 0 <= nei_x < self.size and 0 <= nei_y < self.size and self.current_player == self.board[nei_x][nei_y]:
                    if self.current_player == 1:
                        self.ds_player_1.union((nei_x, nei_y), (x, y))
                    else:
                        self.ds_player_2.union((nei_x, nei_y), (x, y))
            self.move_history.append(move)
            self.current_player *= -1
            if len(self.move_history) >= self.size*2-1:
                self.check_winner()
        else:
            print("Invalid move")
    
    
    def check_winner(self):
        if self.ds_player_1.find(self.top_node) == self.ds_player_1.find(self.bottom_node):
            self.current_winner = 1
            return True
        elif self.ds_player_2.find(self.left_node) == self.ds_player_2.find(self.right_node):
            self.current_winner = -1
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
            print("ties are not possible in hex")

        
    def get_last_move(self):
        return self.move_history[-1]
    
    def is_game_over(self):
        return self.current_winner is not None
    
    #returns a list of valid moves as a hot encoded list
    def hot_encoded_valid_moves(self):
        valid_moves = self.valid_moves()
        hot_encoded_moves = [0]*self.size**2
        for move in valid_moves:
            hot_encoded_moves[move[0]*self.size + move[1]] = 1
        return hot_encoded_moves
    

    #takes in a flattened index and returns the move as a tuple (x, y)
    def index_flatten_moves_to_moves(self, index):
        return (index//self.size, index%self.size)
    
    
    #takes in a move as a tuple (x, y) and returns the flattened index
    def move_to_index_flatten_moves(self, move):
        return move[0]*self.size + move[1]
    
    #returns the board as a binary list. the first two bits represent the current player
    def get_binary_board(self):
        binary_board = []
        if self.current_player == 1:
            binary_board.append(1)
            binary_board.append(0)
        else:
            binary_board.append(0)
            binary_board.append(1)
            
        for row in self.board:
            for cell in row:
                if cell == 1:
                    binary_board.append(1)
                    binary_board.append(0)
                elif cell == -1:
                    binary_board.append(0)
                    binary_board.append(1)
                else:
                    binary_board.append(0)
                    binary_board.append(0)
                    
        return binary_board
    
    
    def print_current_board(self):
        sym={1:'X', -1:'O', 0:'.'}
        n = len(self.board)
        for i in range(n):
            print("  " * (n - i - 1), end="")
            for j in range(i + 1):
                print(sym[self.board[i - j][j]], end="   ")
            print()
        for i in range(n - 2, -1, -1):
            print("  " * (n - i - 1), end="")
            for j in range(i + 1):
                print(sym[self.board[n - j - 1][j + n - i - 1]], end="   ")
            print()
    
    #returns a deep copy of the current game
    def clone(self):
        clone = Hex(self.size)
        clone.board = [[cell for cell in row] for row in self.board]
        clone.cells = self.cells
        clone.top_node = self.top_node
        clone.bottom_node = self.bottom_node
        clone.left_node = self.left_node
        clone.right_node = self.right_node
        
        clone.ds_player_1 = DisjointSet()
        for element, parent in self.ds_player_1._data.items():
            clone.ds_player_1._data[element] = parent
        
        clone.ds_player_2 = DisjointSet()
        for element, parent in self.ds_player_2._data.items():
            clone.ds_player_2._data[element] = parent
        
   
        clone.move_history = [move for move in self.move_history]
        clone.current_player = self.current_player
        clone.current_winner = self.current_winner
        return clone