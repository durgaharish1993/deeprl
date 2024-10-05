
import torch
import torch.nn as nn 
from torch.nn import functional as F 
from config import Parameter

class SimpleModel(nn.Module):

    def __init__(self, config : Parameter):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.state_size, config.hidden_size), #(Batch, state_size)
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.action_size) #(Batch, 128) -> (Batch, num_actions)
        )
    
    def forward(self,x):
        return self.model(x)
