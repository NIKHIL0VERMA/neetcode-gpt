import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        combined = positive + negative
        vocab = sorted({word for s in combined for word in s.split()}) # missing normalization and sanitization (lower and punctuations)
        
        w2id = {word:id+1 for id, word in enumerate(vocab)}
        
        encoded = [torch.tensor([w2id[w] for w in s.split()]) for s in combined]

        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)
