import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = x @ w + b
        y_hat = 1/(1+np.exp(-z))
        error = y_hat - y_true

        loss = 0.5*(error)**2

        delta = (error)*y_hat*(1-y_hat)

        gradw = np.round(delta*x,5)
        gradb = np.round(delta,5)

        return (gradw, gradb)
        
