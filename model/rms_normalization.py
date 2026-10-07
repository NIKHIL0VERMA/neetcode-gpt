import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # convert to numpy vector
        x = np.asarray(x)
        gamma = np.asarray(gamma)

        rms = np.sqrt(np.mean(x**2) + eps)

        x_hat = x/rms

        scaled = gamma*x_hat # output or say y_hat

        return np.round(scaled, 4)
