import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        PE = np.zeros((seq_len, d_model))
        pos = np.arange(seq_len).reshape(-1,1)
        divisor = 10000**(np.arange(0,d_model,2)/d_model)

        PE[:, 0::2] = np.sin(pos/divisor)
        PE[:,1::2] = np.cos(pos/divisor[:PE[:,1::2].shape[1]])
        return np.round(PE, 5)
        
