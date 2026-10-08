import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad() # zero all gradiant
        stats = []
        with torch.no_grad(): # we don't need grad calculation here
            for module in model.children():
                x = module.forward(x)
                if isinstance(module, nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std = round(x.std().item(), 4)
                    if x.dim() > 1:
                        dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    else:
                        dead_frac = round((x<=0).float().mean().item(), 4)
                    stats.append({
                        "std": std,
                        "mean": mean,
                        "dead_fraction": dead_frac
                    })

            return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()
        #forward pass
        y_hat = model.forward(x)
        # calculate MSE loss
        loss = nn.MSELoss()(y_hat, y)
        # backward pass
        loss.backward()

        stats = []
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean = round(grad.mean().item(), 4)
                std = round(grad.std().item(), 4)
                norm = round(torch.norm(grad).item(), 4)
                stats.append({
                    'mean': mean,
                    'std': std,
                    'norm': norm,
                })
        
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for act in activation_stats:
            if act['dead_fraction'] > 0.5:
                return 'dead_neurons'
        
        for grad in gradient_stats:
            if grad['norm'] > 1000:
                return 'exploding_gradients'
        
        if gradient_stats and gradient_stats[-1]['norm'] < 1e-5:
            return 'vanishing_gradients'
        
        for act in activation_stats:
            if act['std'] < 0.1:
                return 'vanishing_gradients'
            if act['std'] > 10.0:
                return 'exploding_gradients'
        
        return 'healthy'
