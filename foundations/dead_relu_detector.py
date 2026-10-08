import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        model.zero_grad() # clear past grad
        dead = []
        with torch.no_grad():
            for module in model.children():
                x = module.forward(x) #forward pass
                if isinstance(module, nn.ReLU):
                    if x.dim() > 1:
                        dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    else:
                        dead_frac = round((x<=0).float().mean().item(), 4)
                    dead.append(dead_frac)

        return dead

    def suggest_fix(self, dead_fractions: List[float]) -> str:

        for frac in dead_fractions:
            if frac > 0.5:
                return 'use_leaky_relu'
        
        if dead_fractions and dead_fractions[0] > 0.3:
            return 'reinitialize'
        
        if len(dead_fractions) >= 2:
            inc = all(
                dead_fractions[i] < dead_fractions[i+1]
                for i in range(len(dead_fractions) - 1)
            )
            if inc and dead_fractions[-1] > 0.1:
                return 'reduce_learning_rate'
        return 'healthy'
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        pass
