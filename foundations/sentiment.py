import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocabulary_size, 16)
        self.lin = nn.Linear(16, 1)
        self.sig = nn.Sigmoid()
        

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        embeded = self.emb(x)
        mn = torch.mean(embeded, dim=1)
        z = self.lin(mn)
        y_hat = self.sig(z)
        return torch.round(y_hat, decimals=4)