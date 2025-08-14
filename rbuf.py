import random

class RBUF:
    
    def __init__(self):
        self.buffer = []
    
    def add(self, state):
        self.buffer.append(state)
    
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return
        return random.sample(self.buffer, batch_size)  
        
