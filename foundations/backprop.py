import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float):
        return np.dot(x, w) + b

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = self.forward(x, w, b)
        y_hat = self.sigmoid(z)
        delta = (y_hat - y_true)*y_hat*(1-y_hat)
        dl_dw = delta*x

        return (np.round(dl_dw, 5), np.round(delta, 5))

        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        