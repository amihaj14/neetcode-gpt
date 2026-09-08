import torch
from typing import List, Tuple


class Solution:
    def batch_loader(
        self, raw_dataset: str, context_length: int, batch_size: int
    ) -> Tuple[List[List[str]], List[List[str]]]:

        torch.manual_seed(0)
        
        tokens = raw_dataset.split()
        start_i = torch.randint(len(tokens) - context_length, (batch_size,))
        
        X = [tokens[i:i+context_length] for i in start_i]
        Y = [tokens[i+1: i+1+context_length] for i in start_i]

        return X,Y