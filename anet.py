import torch
import torch.nn as nn
import torch.optim as optim
from config import anet_config




class ANET(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.loss_history=[]
        self.loss_function=nn.CrossEntropyLoss()
        self.input_size=anet_config["input_size"]
        self.output_size=anet_config["output_size"]
        self.hidden_layers=anet_config["hidden_layers"]
        self.activation_functions=anet_config["activation"]
        
        self.stack=self.construct_layers()

        optimizer_class = getattr(optim, anet_config["optimizer"])
        self.optimizer = optimizer_class(self.parameters(), lr=anet_config["lr"])


    def construct_layers(self):
        layers=[]
        layers.append(nn.Linear(self.input_size, self.hidden_layers[0]))
        layers.append(self.get_activation_function(self.activation_functions[0]))
        layers.append(nn.BatchNorm1d(self.hidden_layers[0]))
        
        for layer_number in range (len(self.hidden_layers)-1):
            layers.append(nn.Linear(self.hidden_layers[layer_number], self.hidden_layers[layer_number+1]))
            layers.append(self.get_activation_function(self.activation_functions[layer_number+1]))
            layers.append(nn.BatchNorm1d(self.hidden_layers[layer_number+1]))
        layers.append(nn.Linear(self.hidden_layers[-1], self.output_size))
        return nn.Sequential(*layers)
    
    
    def get_activation_function(self, activation):
        if activation=="relu":
            return nn.ReLU()
        elif activation=="sigmoid":
            return nn.Sigmoid()
        elif activation=="tanh":
            return nn.Tanh()
        elif activation=="linear":
            return nn.Identity()
        
        
    def forward(self, x):
        if isinstance(x, list):
           x=torch.tensor([x], dtype=torch.float32)
        logits = self.stack(x)
        return logits
    

    
    def train_model(self, training_data):
        self.train()
        
        inputs, labels = zip(*training_data)
    
        inputs=torch.tensor(inputs, dtype=torch.float32)
        labels=torch.tensor(labels, dtype=torch.float32)

        self.optimizer.zero_grad()
        
        outputs = self(inputs)
    
        loss = self.loss_function(outputs, labels)  
        self.loss_history.append(loss.item())
        
        loss.backward()
        self.optimizer.step()
    

