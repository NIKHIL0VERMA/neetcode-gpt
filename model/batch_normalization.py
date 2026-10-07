import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float], running_mean: List[float], running_var: List[float], momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        x = np.asarray(x)
        gamma = np.asarray(gamma)
        beta = np.asarray(beta)
        running_mean = np.asarray(running_mean)
        running_var = np.asarray(running_var)

        if training:
            mu = np.mean(x, axis=0)
            var = np.mean((x-mu)**2, axis=0)
            x_hat = (x-mu)/np.sqrt(eps+var)
            running_mean = (1-momentum)*running_mean + mu*momentum
            running_var = (1-momentum)*running_var + var*momentum
        else:
            x_hat = (x-running_mean)/np.sqrt(running_var+eps)
        scaled = x_hat*gamma
        y_hat = scaled + beta

        return (np.round(y_hat, 4).tolist(), np.round(running_mean, 4).tolist(), np.round(running_var, 4).tolist())