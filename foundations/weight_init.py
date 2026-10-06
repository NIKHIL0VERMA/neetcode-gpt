import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0/(fan_in+fan_out))
        weights = torch.randn(fan_out, fan_in)*std
        return torch.round(weights, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        std = math.sqrt(2.0/fan_in)
        weights = torch.randn(fan_out, fan_in)*std
        return torch.round(weights, decimals=4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        
        # create dimensions to form layers
        # input_dim is starting dimension we have to init the random matrix from this dim then we have few hidden layers
        dims = [input_dim] + [hidden_dim]*num_layers

        # init weights, starting with input_dim till end
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0/(dims[i] + dims[i+1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0/dims[i])
            else:
                std = 1
            # record the weights
            weights.append(torch.randn(dims[i+1], dims[i])*std)

        # init random input features
        input = torch.randn(1, input_dim)
        
        # calculate the std at each layer
        stds = []
        for w in weights:
            input = torch.relu(input@w.T)
            stds.append(round(input.std().item(), 2))

        return stds