import torch
import torch.nn as nn
import torch.nn.functional as F

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:

        optimizer = torch.optim.AdamW(model.parameters(), lr)

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            start_i = torch.randint(len(data) - context_length, (batch_size,))

            X = torch.stack([data[i:i+context_length] for i in start_i])
            Y = torch.stack([data[i+1: i+1+context_length] for i in start_i])

            logits = model(X)
            logits_flat = logits.reshape(logits.shape[0]*logits.shape[1], logits.shape[-1])
            Y_flat = Y.reshape(logits.shape[0]*logits.shape[1])

            loss = F.cross_entropy(logits_flat, Y_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)