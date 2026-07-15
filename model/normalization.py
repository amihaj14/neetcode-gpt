import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        eps = 1e-5
        mean = np.sum(x)/len(x) 
        var = np.sum((x-mean)**2)/len(x)

        x_hat = (x - mean)/np.sqrt(var+eps)
        scale_shift = gamma*x_hat + beta

        return np.round(scale_shift ,5)
