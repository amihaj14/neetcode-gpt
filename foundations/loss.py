import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        n = len(y_true)
        y_pred = y_pred + eps
        true = y_true*np.log(y_pred)
        false = (1-y_true)*np.log(1-y_pred)
        return np.round((-1/n)*np.sum(true+false),4)

        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        eps = 1e-7
        n = len(y_true)
        y_pred = y_pred + eps
        loss = y_true*np.log(y_pred)
        return np.round((-1/n)*np.sum(loss),4)
        
