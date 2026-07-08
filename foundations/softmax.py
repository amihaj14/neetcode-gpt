import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        m = np.max(z)
        exp = np.exp(z-m)
        return np.round(exp/np.sum(exp), 4)

