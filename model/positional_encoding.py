import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        pose = np.zeros((seq_len, d_model))
        position = np.arange(seq_len).reshape(-1,1)
        divi = 10000**(np.arange(0, d_model, 2)/d_model)
        pose[:, 0::2] = np.sin(position/divi)
        pose[:, 1::2] = np.cos(position/divi)
        return np.round(pose, 5)