import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        
        n = X.shape[0]
        w = np.zeros(X.shape[1])
        b = 0.0

        for i in range(epochs):
            y_hat = X @ w + b
            loss = np.sum((y_hat - y)**2)/n

            dLdw = (X.T @ (y_hat - y)) * (2/n)
            dLdb = np.sum(y_hat - y) * (2/n)

            w = w - lr*dLdw
            b = b - lr*dLdb

        return (np.round(w, decimals=5), np.round(b, decimals=5))
