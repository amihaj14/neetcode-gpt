import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats=[]

        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean = round(x.mean().item(),4)
                    std = round(x.std().item(), 4)
                    dead_frac = (x <=0).all(dim=0).float().mean().item()

                    stats.append({'mean': mean, 'std': std, 'dead_fraction': dead_frac})
        return stats
        

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()
        criterion = nn.MSELoss()
        output = model(x)
        loss = criterion(output,y)
        loss.backward()

        stats =[]
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                stats.append({"mean": round(grad.mean().item(),4), "std": round(grad.std().item(), 4), "norm": round(torch.norm(grad).item(), 4)}) 

        return stats


    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        if any(s["dead_fraction"] > 0.5 for s in activation_stats):
            return "dead_neurons"

        if any(s["norm"] > 1000 for s in gradient_stats):
            return "exploding_gradients"

        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        if any(s["std"] < 0.1 for s in activation_stats):
            return "vanishing_gradients"

        if any(s["std"] > 10.0 for s in activation_stats):
            return "exploding_gradients"

        return 'healthy'