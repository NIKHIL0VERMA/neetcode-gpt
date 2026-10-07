import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # init weights and bias
        b = 0.0
        W = np.zeros(X.shape[1])

        for _ in range(epochs):
            # predict
            y_hat = X@W + b

            # calculate mse loss
            loss = np.mean((y_hat - y)**2)

            # backpropagate grad
            # d_loss/d_y_hat
            dl = 2*(y_hat-y)/len(y)

            # d_loss/d_weight = (d_loss/d_y_hat)*(d_y_hat/d_weight)
            dw = X.T@dl

            #d_loss/d_b = (d_loss/d_y_hat)*(d_y_hat/d_b) = sum(dl)
            db = np.sum(dl)

            # update weights
            W = W - lr*dw

            # update bias
            b = b - lr*db

        return (np.round(W, 5), np.round(b, 5))