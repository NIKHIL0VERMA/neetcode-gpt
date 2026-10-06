import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        inputs = x
        for i in range(len(biases)):
            inputs = inputs@weights[i] + biases[i]
            if i < len(biases) -1:
                inputs = np.maximum(0, inputs)
        return np.round(inputs, 5)
