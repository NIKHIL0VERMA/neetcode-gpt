import numpy as np
from numpy.typing import NDArray


class Solution:
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def relu(self, x):
        return max(0.0, x)

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        summ = np.dot(x,w) + b
        if activation == 'relu':
            out = self.relu(summ)
        elif activation == 'sigmoid':
            out = self.sigmoid(summ)
        else:
            out = summ
        
        return np.round(float(out), 5)
