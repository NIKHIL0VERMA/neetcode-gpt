import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        normalize = z - np.max(z, axis=-1)
        ex = np.exp(normalize)
        ans = ex/np.sum(ex)
        return np.round(ans, 4)
