"""
Residual Network implementation for empirical verification of near-orthogonality theorem.
Implements L-layer residual network with configurable width and depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Single residual block F_l(x; theta_l)."""
    
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.linear = nn.Linear(width, width)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.linear(x))
    
    def jacobian(self, x):
        """Compute Jacobian J_{F_l} = ∂F_l/∂x."""
        x = x.detach().requires_grad_(True)
        out = self.forward(x)
        batch_size = x.shape[0]
        
        # Compute Jacobian for each sample in batch
        jacobians = []
        for i in range(batch_size):
            grad = torch.autograd.grad(
                out[i].sum(dim=0), 
                x, 
                retain_graph=True,
                create_graph=False
            )[0][i]  # [width]
            jacobians.append(grad)
        
        return torch.stack(jacobians)  # [batch_size, width]


class ResNet(nn.Module):
    """L-layer residual network."""
    
    def __init__(self, L, width, input_dim, output_dim):
        super().__init__()
        self.L = L
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, width)
        
        # Residual blocks
        self.blocks = nn.ModuleList([ResidualBlock(width) for _ in range(L)])
        
        # Output projection
        self.output_proj = nn.Linear(width, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = x + block(x)
        
        return self.output_proj(x)
    
    def get_layer_outputs(self, x):
        """Get intermediate layer outputs for pushforward computation."""
        x = self.input_proj(x)
        outputs = [x]
        
        for block in self.blocks:
            x = x + block(x)
            outputs.append(x)
        
        return outputs
    
    def initialize_he(self):
        """Initialize with He initialization (variance = 2/W)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
