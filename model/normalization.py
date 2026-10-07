import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        mu = np.mean(x)
        var = np.mean((x-mu)**2)
        eps = 1e-5
        
        x_hat = (x-mu)/math.sqrt(var+eps)
        scale = gamma*x_hat
        shift = scale + beta

        return np.round(shift, 5)