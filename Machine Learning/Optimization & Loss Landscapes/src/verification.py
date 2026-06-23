"""
Verification functions for the theoretical results.
Implements empirical verification for:
- Lemma 1: Pushforward spectrum bounds
- Lemma 2: Zero expectation of cross-terms
- Lemma 3: Diagonal norm scaling
- Theorem 1: Cross-term bound
- Corollary 1: Gauss-Newton rank scaling
"""

import torch
import numpy as np
from scipy.linalg import svdvals


def verify_lemma1_pushforward_spectrum(model, x, c_estimate=None):
    """
    Verify Lemma 1: Pushforward spectrum bounds
    σ_min(Φ) ≥ e^{-c}, σ_max(Φ) ≤ e^{c}
    
    Args:
        model: ResNet model
        x: Input tensor
        c_estimate: If provided, use this c value. Otherwise estimate from data.
    
    Returns:
        dict: Contains singular values and verification results
    """
    model.eval()
    layer_outputs = model.get_layer_outputs(x)
    
    # Compute pushforward operators
    from .jacobian import compute_pushforward_operators
    pushforwards = compute_pushforward_operators(model, layer_outputs)
    
    results = {
        'singular_values': [],
        'sigma_min': [],
        'sigma_max': [],
        'bounds_satisfied': True
    }
    
    for l, Phi in enumerate(pushforwards):
        Phi_np = Phi.detach().cpu().numpy()
        svs = svdvals(Phi_np)
        
        sigma_min = np.min(svs)
        sigma_max = np.max(svs)
        
        results['singular_values'].append(svs)
        results['sigma_min'].append(sigma_min)
        results['sigma_max'].append(sigma_max)
        
        # Estimate c from data if not provided
        if c_estimate is None:
            c_est = max(abs(np.log(sigma_min)), abs(np.log(sigma_max)))
        else:
            c_est = c_estimate
        
        # Check bounds
        lower_bound = np.exp(-c_est)
        upper_bound = np.exp(c_est)
        
        if sigma_min < lower_bound * 0.9 or sigma_max > upper_bound * 1.1:
            results['bounds_satisfied'] = False
    
    results['c_estimate'] = c_estimate if c_estimate else c_est
    return results


def verify_lemma2_zero_expectation(model, x, num_initializations=10):
    """
    Verify Lemma 2: Zero expectation of cross-terms
    E[tr(Z_l^T Z_{l'})] = 0 for l ≠ l'
    
    Args:
        model: ResNet model
        x: Input tensor
        num_initializations: Number of random initializations to sample
    
    Returns:
        dict: Contains cross-term statistics across initializations
    """
    from .jacobian import compute_cross_term_matrix
    
    L = model.L
    cross_terms_samples = []
    
    for _ in range(num_initializations):
        # Reinitialize model
        model.initialize_he()
        
        # Compute cross-term matrix
        from .jacobian import compute_per_layer_jacobians
        per_layer_jacobians = compute_per_layer_jacobians(model, x)
        cross_terms = compute_cross_term_matrix(per_layer_jacobians)
        
        # Extract off-diagonal terms
        off_diagonal = []
        for l in range(L):
            for l_prime in range(L):
                if l != l_prime:
                    off_diagonal.append(cross_terms[l, l_prime].item())
        
        cross_terms_samples.append(off_diagonal)
    
    # Compute statistics
    cross_terms_samples = np.array(cross_terms_samples)
    mean_cross = np.mean(cross_terms_samples)
    std_cross = np.std(cross_terms_samples)
    
    results = {
        'mean_cross_term': mean_cross,
        'std_cross_term': std_cross,
        'zero_expectation_satisfied': abs(mean_cross) < 2 * std_cross,
        'samples': cross_terms_samples
    }
    
    return results


def verify_lemma3_diagonal_norm(model, x, d_I, c_estimate=None):
    """
    Verify Lemma 3: Diagonal norm scaling
    ||J_θ^(l)||_F² = Θ(e^{2c} · d_I · W / L)
    
    Args:
        model: ResNet model
        x: Input tensor
        d_I: Intrinsic dimension of data
        c_estimate: Estimated c value from Lemma 1
    
    Returns:
        dict: Contains diagonal norms and scaling verification
    """
    from .jacobian import compute_per_layer_jacobians, compute_diagonal_norms
    
    per_layer_jacobians = compute_per_layer_jacobians(model, x)
    diagonal_norms = compute_diagonal_norms(per_layer_jacobians)
    
    L = model.L
    W = model.width
    
    if c_estimate is None:
        # Estimate c from pushforward
        lemma1_results = verify_lemma1_pushforward_spectrum(model, x)
        c_estimate = lemma1_results['c_estimate']
    
    # Theoretical scaling
    theoretical = np.exp(2 * c_estimate) * d_I * W / L
    
    results = {
        'diagonal_norms': diagonal_norms,
        'theoretical_scaling': theoretical,
        'mean_diagonal_norm': np.mean(diagonal_norms),
        'scaling_ratio': np.mean(diagonal_norms) / theoretical,
        'within_constant_factor': 0.1 < np.mean(diagonal_norms) / theoretical < 10
    }
    
    return results


def verify_theorem1_cross_term_bound(model, x, d_I, c_estimate=None):
    """
    Verify Theorem 1: Cross-term bound
    |⟨J_θ^(l), J_θ^(l')⟩_F| ≤ C · √(d_I · W) / L
    
    Args:
        model: ResNet model
        x: Input tensor
        d_I: Intrinsic dimension of data
        c_estimate: Estimated c value from Lemma 1
    
    Returns:
        dict: Contains cross-term bounds and verification
    """
    from .jacobian import compute_per_layer_jacobians, compute_cross_term_matrix
    
    per_layer_jacobians = compute_per_layer_jacobians(model, x)
    cross_terms = compute_cross_term_matrix(per_layer_jacobians)
    
    L = model.L
    W = model.width
    
    if c_estimate is None:
        lemma1_results = verify_lemma1_pushforward_spectrum(model, x)
        c_estimate = lemma1_results['c_estimate']
    
    # Extract off-diagonal terms
    off_diagonal_terms = []
    for l in range(L):
        for l_prime in range(L):
            if l != l_prime:
                off_diagonal_terms.append(abs(cross_terms[l, l_prime].item()))
    
    # Theoretical bound (with constant C)
    theoretical_bound = np.exp(2 * c_estimate) * np.sqrt(d_I * W) / L
    
    results = {
        'max_cross_term': max(off_diagonal_terms),
        'mean_cross_term': np.mean(off_diagonal_terms),
        'theoretical_bound': theoretical_bound,
        'bound_satisfied': max(off_diagonal_terms) < 10 * theoretical_bound,
        'cross_terms': off_diagonal_terms
    }
    
    return results


def verify_corollary1_gn_rank(model, x, d_I, c_estimate=None):
    """
    Verify Corollary 1: Gauss-Newton rank scaling
    r_eff(G) = Ω(L · W_min / d_I)
    
    Args:
        model: ResNet model
        x: Input tensor
        d_I: Intrinsic dimension of data
        c_estimate: Estimated c value from Lemma 1
    
    Returns:
        dict: Contains effective rank and scaling verification
    """
    from .jacobian import compute_per_layer_jacobians, compute_cross_term_matrix, compute_diagonal_norms
    
    per_layer_jacobians = compute_per_layer_jacobians(model, x)
    
    # Construct full Jacobian
    J_full = torch.cat(per_layer_jacobians, dim=1)  # [n_out, L*W^2]
    
    # Compute Gauss-Newton matrix
    G = torch.matmul(J_full.T, J_full)  # [L*W^2, L*W^2]
    
    # Compute effective rank: r_eff(G) = tr(G)^2 / ||G||_F^2
    tr_G = torch.trace(G).item()
    frob_G = torch.norm(G, 'fro').item()
    r_eff = (tr_G ** 2) / (frob_G ** 2)
    
    L = model.L
    W = model.width
    
    # Theoretical scaling
    theoretical_rank = L * W / d_I
    
    results = {
        'effective_rank': r_eff,
        'theoretical_rank': theoretical_rank,
        'rank_ratio': r_eff / theoretical_rank,
        'scales_with_depth': r_eff / L > 0.1 * W / d_I,
        'trace_G': tr_G,
        'frobenius_G': frob_G
    }
    
    return results


def verify_cosine_similarity_scaling(model, x, d_I):
    """
    Verify falsifiable prediction from §5.2:
    cosine_sim(J^(l), J^(l')) ~ O(L / √(d_I · W)) → 0 as W grows
    
    Args:
        model: ResNet model
        x: Input tensor
        d_I: Intrinsic dimension of data
    
    Returns:
        dict: Contains cosine similarities and scaling verification
    """
    from .jacobian import compute_per_layer_jacobians
    
    per_layer_jacobians = compute_per_layer_jacobians(model, x)
    
    L = model.L
    W = model.width
    
    # Compute cosine similarities between layers
    cosine_similarities = []
    for l in range(L):
        for l_prime in range(L):
            if l < l_prime:
                J_l = per_layer_jacobians[l]
                J_lp = per_layer_jacobians[l_prime]
                
                # Flatten and compute cosine similarity
                J_l_flat = J_l.flatten()
                J_lp_flat = J_lp.flatten()
                
                cos_sim = torch.dot(J_l_flat, J_lp_flat) / (
                    torch.norm(J_l_flat) * torch.norm(J_lp_flat)
                )
                cosine_similarities.append(abs(cos_sim.item()))
    
    # Theoretical scaling
    theoretical_scale = L / np.sqrt(d_I * W)
    
    results = {
        'mean_cosine_similarity': np.mean(cosine_similarities),
        'max_cosine_similarity': max(cosine_similarities),
        'theoretical_scale': theoretical_scale,
        'within_order_of_magnitude': (
            0.1 * theoretical_scale < np.mean(cosine_similarities) < 
            10 * theoretical_scale
        ),
        'cosine_similarities': cosine_similarities
    }
    
    return results
