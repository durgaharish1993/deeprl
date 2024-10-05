from dataclasses import dataclass

@dataclass
class Parameter:
    state_size : int =  10
    action_size : int = 5
    batch_size : int =  3
    hidden_size : int = 128



