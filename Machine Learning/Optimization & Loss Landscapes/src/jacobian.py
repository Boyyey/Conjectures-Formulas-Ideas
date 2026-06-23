"""
Jacobian computation utilities for per-layer parameter Jacobians.
Computes J_θ^(l) = Φ_{L←l} · Z_l for each layer l.
"""

import torch
import torch.nn as nn


def compute_per_layer_jacobians(model, x, output_idx=None):
    """
    Compute per-layer parameter Jacobians J_θ^(l) for all layers.
    
    Args:
        model: ResNet model
        x: Input tensor [batch_size, input_dim]
        output_idx: If specified, compute Jacobian w.r.t. specific output dimension
    
    Returns:
        List of per-layer Jacobians, each of shape [n_out, p_l]
        where p_l is the number of parameters in layer l
    """
    model.eval()
    batch_size = x.shape[0]
    
    # Get layer outputs for pushforward computation
    layer_outputs = model.get_layer_outputs(x)
    
    # Compute pushforward operators Φ_{L←l}
    pushforwards = compute_pushforward_operators(model, layer_outputs)
    
    # Compute per-layer Jacobians
    per_layer_jacobians = []
    
    for l in range(model.L):
        # Get parameters for layer l
        block = model.blocks[l]
        params = list(block.parameters())
        
        if len(params) == 0:
            continue
            
        # Flatten all parameters for this layer
        param_vec = torch.cat([p.view(-1) for p in params])
        p_l = param_vec.shape[0]
        
        # Compute Z_l = ∂x_l/∂θ_l
        # This is the local parameter Jacobian
        Z_l = compute_local_jacobian(block, layer_outputs[l], param_vec)
        
        # Apply pushforward: J_θ^(l) = Φ_{L←l} · Z_l
        Phi = pushforwards[l]
        J_l = torch.matmul(Phi, Z_l)
        
        per_layer_jacobians.append(J_l)
    
    return per_layer_jacobians


def compute_local_jacobian(block, x_input, param_vec):
    """
    Compute local parameter Jacobian Z_l = ∂x_l/∂θ_l.
    
    Args:
        block: Residual block
        x_input: Input to the block [batch_size, width]
        param_vec: Flattened parameter vector
    
    Returns:
        Z_l: Local Jacobian [batch_size * width, p_l]
    """
    batch_size, width = x_input.shape
    p_l = param_vec.shape[0]
    
    # Compute Jacobian via autograd
    x_input = x_input.detach().requires_grad_(True)
    output = block(x_input)
    
    # Flatten output for gradient computation
    output_flat = output.view(-1)
    
    # Compute gradient w.r.t. input (this gives us the sensitivity)
    # For parameter Jacobian, we need gradient w.r.t. parameters
    Z_l = torch.zeros(batch_size * width, p_l)
    
    params = list(block.parameters())
    param_idx = 0
    
    for i in range(batch_size * width):
        grad_output = torch.zeros_like(output_flat)
        grad_output[i] = 1.0
        
        # Compute gradients w.r.t. parameters
        grads = torch.autograd.grad(
            output_flat, 
            params, 
            grad_outputs=grad_output,
            retain_graph=(i < batch_size * width - 1)
        )
        
        # Concatenate gradients
        grad_vec = torch.cat([g.view(-1) for g in grads])
        Z_l[i] = grad_vec
    
    return Z_l


def compute_pushforward_operators(model, layer_outputs):
    """
    Compute pushforward operators Φ_{L←l} = ∏_{k=l}^{L-1} (I + J_{F_k}).
    
    Args:
        model: ResNet model
        layer_outputs: List of layer outputs
    
    Returns:
        List of pushforward operators [Φ_{L←l}] for each layer l
    """
    L = model.L
    pushforwards = []
    
    # Compute Jacobians of each residual block
    block_jacobians = []
    for l in range(L):
        block = model.blocks[l]
        x_l = layer_outputs[l]
        J_F = block.jacobian(x_l)  # [batch_size, width]
        block_jacobians.append(J_F)
    
    # Compute pushforward for each layer
    for l in range(L):
        Phi = torch.eye(model.width)
        
        for k in range(l, L):
            J_F = block_jacobians[k]
            # Phi = Phi · (I + J_F)
            # For simplicity, use identity + mean Jacobian
            J_F_mean = J_F.mean(dim=0)  # [width]
            Phi = Phi @ (torch.eye(model.width) + torch.diag(J_F_mean))
        
        pushforwards.append(Phi)
    
    return pushforwards


def compute_cross_term_matrix(per_layer_jacobians):
    """
    Compute cross-term matrix G_{ll'} = (J_θ^(l))^T J_θ^(l').
    
    Args:
        per_layer_jacobians: List of per-layer Jacobians
    
    Returns:
        Cross-term matrix [L, L] where each entry is the Frobenius inner product
    """
    L = len(per_layer_jacobians)
    cross_terms = torch.zeros(L, L)
    
    for l in range(L):
        for l_prime in range(L):
            J_l = per_layer_jacobians[l]
            J_lp = per_layer_jacobians[l_prime]
            
            # Frobenius inner product: tr(J_l^T J_lp)
            inner_product = torch.sum(J_l * J_lp)
            cross_terms[l, l_prime] = inner_product
    
    return cross_terms


def compute_diagonal_norms(per_layer_jacobians):
    """
    Compute diagonal norms ||J_θ^(l)||_F² for each layer.
    
    Args:
        per_layer_jacobians: List of per-layer Jacobians
    
    Returns:
        List of Frobenius norms squared for each layer
    """
    diagonal_norms = []
    
    for J_l in per_layer_jacobians:
        norm_sq = torch.sum(J_l ** 2)
        diagonal_norms.append(norm_sq.item())
    
    return diagonal_norms
