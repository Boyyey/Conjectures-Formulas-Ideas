"""
Main experiment script for empirical verification of near-orthogonality theorem.
Runs experiments across different depths, widths, and intrinsic dimensions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

from src.resnet import ResNet
from src.jacobian import compute_per_layer_jacobians, compute_cross_term_matrix, compute_diagonal_norms
from src.verification import (
    verify_lemma1_pushforward_spectrum,
    verify_lemma2_zero_expectation,
    verify_lemma3_diagonal_norm,
    verify_theorem1_cross_term_bound,
    verify_corollary1_gn_rank,
    verify_cosine_similarity_scaling
)


def generate_synthetic_data(batch_size, input_dim, d_I):
    """
    Generate synthetic data with intrinsic dimension d_I.
    
    Args:
        batch_size: Number of samples
        input_dim: Full input dimension
        d_I: Intrinsic dimension (d_I <= input_dim)
    
    Returns:
        Data tensor of shape [batch_size, input_dim]
    """
    # Generate random low-dimensional data
    Z = torch.randn(batch_size, d_I)
    
    # Random projection to full dimension
    P = torch.randn(input_dim, d_I)
    
    # Add noise in orthogonal directions
    noise = torch.randn(batch_size, input_dim - d_I) * 0.01
    
    # Combine
    data = torch.matmul(Z, P.T)
    data = torch.cat([data, noise], dim=1)
    
    return data


def run_single_experiment(L, W, input_dim, d_I, batch_size=32):
    """
    Run a single experiment with given architecture parameters.
    
    Args:
        L: Number of layers
        W: Width
        input_dim: Input dimension
        d_I: Intrinsic dimension
        batch_size: Batch size
    
    Returns:
        dict: Experiment results
    """
    # Create model
    model = ResNet(L=L, width=W, input_dim=input_dim, output_dim=10)
    model.initialize_he()
    
    # Generate data
    x = generate_synthetic_data(batch_size, input_dim, d_I)
    
    # Run verifications
    results = {
        'L': L,
        'W': W,
        'd_I': d_I,
    }
    
    # Lemma 1: Pushforward spectrum
    lemma1 = verify_lemma1_pushforward_spectrum(model, x)
    results['lemma1'] = lemma1
    c_estimate = lemma1['c_estimate']
    
    # Lemma 2: Zero expectation
    lemma2 = verify_lemma2_zero_expectation(model, x, num_initializations=5)
    results['lemma2'] = lemma2
    
    # Lemma 3: Diagonal norm
    lemma3 = verify_lemma3_diagonal_norm(model, x, d_I, c_estimate)
    results['lemma3'] = lemma3
    
    # Theorem 1: Cross-term bound
    theorem1 = verify_theorem1_cross_term_bound(model, x, d_I, c_estimate)
    results['theorem1'] = theorem1
    
    # Corollary 1: GN rank
    corollary1 = verify_corollary1_gn_rank(model, x, d_I, c_estimate)
    results['corollary1'] = corollary1
    
    # Cosine similarity scaling
    cosine = verify_cosine_similarity_scaling(model, x, d_I)
    results['cosine_similarity'] = cosine
    
    return results


def run_width_sweep():
    """Run experiments across different widths."""
    print("Running width sweep...")
    
    L = 10
    input_dim = 100
    d_I = 10
    widths = [32, 64, 128, 256, 512]
    
    results = []
    for W in tqdm(widths):
        result = run_single_experiment(L, W, input_dim, d_I)
        results.append(result)
    
    # Plot results
    plot_width_sweep(results, widths)
    return results


def run_depth_sweep():
    """Run experiments across different depths."""
    print("Running depth sweep...")
    
    W = 128
    input_dim = 100
    d_I = 10
    depths = [5, 10, 20, 30, 50]
    
    results = []
    for L in tqdm(depths):
        result = run_single_experiment(L, W, input_dim, d_I)
        results.append(result)
    
    # Plot results
    plot_depth_sweep(results, depths)
    return results


def run_dI_sweep():
    """Run experiments across different intrinsic dimensions."""
    print("Running d_I sweep...")
    
    L = 10
    W = 128
    input_dim = 100
    d_I_values = [5, 10, 20, 30, 50]
    
    results = []
    for d_I in tqdm(d_I_values):
        result = run_single_experiment(L, W, input_dim, d_I)
        results.append(result)
    
    # Plot results
    plot_dI_sweep(results, d_I_values)
    return results


def plot_width_sweep(results, widths):
    """Plot width sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cross-term vs width
    cross_terms = [r['theorem1']['mean_cross_term'] for r in results]
    theoretical = [r['theorem1']['theoretical_bound'] for r in results]
    
    axes[0, 0].plot(widths, cross_terms, 'o-', label='Empirical')
    axes[0, 0].plot(widths, theoretical, '--', label='Theoretical (O(√W))')
    axes[0, 0].set_xlabel('Width (W)')
    axes[0, 0].set_ylabel('Cross-term magnitude')
    axes[0, 0].set_title('Theorem 1: Cross-term bound')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    
    # Cosine similarity vs width
    cosine_sims = [r['cosine_similarity']['mean_cosine_similarity'] for r in results]
    theoretical_cos = [r['cosine_similarity']['theoretical_scale'] for r in results]
    
    axes[0, 1].plot(widths, cosine_sims, 'o-', label='Empirical')
    axes[0, 1].plot(widths, theoretical_cos, '--', label='Theoretical (O(1/√W))')
    axes[0, 1].set_xlabel('Width (W)')
    axes[0, 1].set_ylabel('Cosine similarity')
    axes[0, 1].set_title('§5.2: Cosine similarity scaling')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # Effective rank vs width
    ranks = [r['corollary1']['effective_rank'] for r in results]
    theoretical_rank = [r['corollary1']['theoretical_rank'] for r in results]
    
    axes[1, 0].plot(widths, ranks, 'o-', label='Empirical')
    axes[1, 0].plot(widths, theoretical_rank, '--', label='Theoretical (O(W))')
    axes[1, 0].set_xlabel('Width (W)')
    axes[1, 0].set_ylabel('Effective rank')
    axes[1, 0].set_title('Corollary 1: GN rank scaling')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # Diagonal norm vs width
    diag_norms = [np.mean(r['lemma3']['diagonal_norms']) for r in results]
    theoretical_diag = [r['lemma3']['theoretical_scaling'] for r in results]
    
    axes[1, 1].plot(widths, diag_norms, 'o-', label='Empirical')
    axes[1, 1].plot(widths, theoretical_diag, '--', label='Theoretical (O(W))')
    axes[1, 1].set_xlabel('Width (W)')
    axes[1, 1].set_ylabel('Diagonal norm')
    axes[1, 1].set_title('Lemma 3: Diagonal norm scaling')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('width_sweep_results.png', dpi=150)
    plt.close()


def plot_depth_sweep(results, depths):
    """Plot depth sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cross-term vs depth
    cross_terms = [r['theorem1']['mean_cross_term'] for r in results]
    theoretical = [r['theorem1']['theoretical_bound'] for r in results]
    
    axes[0, 0].plot(depths, cross_terms, 'o-', label='Empirical')
    axes[0, 0].plot(depths, theoretical, '--', label='Theoretical (O(1/L))')
    axes[0, 0].set_xlabel('Depth (L)')
    axes[0, 0].set_ylabel('Cross-term magnitude')
    axes[0, 0].set_title('Theorem 1: Cross-term bound')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    
    # Cosine similarity vs depth
    cosine_sims = [r['cosine_similarity']['mean_cosine_similarity'] for r in results]
    theoretical_cos = [r['cosine_similarity']['theoretical_scale'] for r in results]
    
    axes[0, 1].plot(depths, cosine_sims, 'o-', label='Empirical')
    axes[0, 1].plot(depths, theoretical_cos, '--', label='Theoretical (O(L))')
    axes[0, 1].set_xlabel('Depth (L)')
    axes[0, 1].set_ylabel('Cosine similarity')
    axes[0, 1].set_title('§5.2: Cosine similarity scaling')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # Effective rank vs depth
    ranks = [r['corollary1']['effective_rank'] for r in results]
    theoretical_rank = [r['corollary1']['theoretical_rank'] for r in results]
    
    axes[1, 0].plot(depths, ranks, 'o-', label='Empirical')
    axes[1, 0].plot(depths, theoretical_rank, '--', label='Theoretical (O(L))')
    axes[1, 0].set_xlabel('Depth (L)')
    axes[1, 0].set_ylabel('Effective rank')
    axes[1, 0].set_title('Corollary 1: GN rank scaling')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # Diagonal norm vs depth
    diag_norms = [np.mean(r['lemma3']['diagonal_norms']) for r in results]
    theoretical_diag = [r['lemma3']['theoretical_scaling'] for r in results]
    
    axes[1, 1].plot(depths, diag_norms, 'o-', label='Empirical')
    axes[1, 1].plot(depths, theoretical_diag, '--', label='Theoretical (O(1/L))')
    axes[1, 1].set_xlabel('Depth (L)')
    axes[1, 1].set_ylabel('Diagonal norm')
    axes[1, 1].set_title('Lemma 3: Diagonal norm scaling')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('depth_sweep_results.png', dpi=150)
    plt.close()


def plot_dI_sweep(results, d_I_values):
    """Plot d_I sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Cross-term vs d_I
    cross_terms = [r['theorem1']['mean_cross_term'] for r in results]
    theoretical = [r['theorem1']['theoretical_bound'] for r in results]
    
    axes[0, 0].plot(d_I_values, cross_terms, 'o-', label='Empirical')
    axes[0, 0].plot(d_I_values, theoretical, '--', label='Theoretical (O(√d_I))')
    axes[0, 0].set_xlabel('Intrinsic dimension (d_I)')
    axes[0, 0].set_ylabel('Cross-term magnitude')
    axes[0, 0].set_title('Theorem 1: Cross-term bound')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_yscale('log')
    
    # Cosine similarity vs d_I
    cosine_sims = [r['cosine_similarity']['mean_cosine_similarity'] for r in results]
    theoretical_cos = [r['cosine_similarity']['theoretical_scale'] for r in results]
    
    axes[0, 1].plot(d_I_values, cosine_sims, 'o-', label='Empirical')
    axes[0, 1].plot(d_I_values, theoretical_cos, '--', label='Theoretical (O(√d_I))')
    axes[0, 1].set_xlabel('Intrinsic dimension (d_I)')
    axes[0, 1].set_ylabel('Cosine similarity')
    axes[0, 1].set_title('§5.2: Cosine similarity scaling')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_yscale('log')
    
    # Effective rank vs d_I
    ranks = [r['corollary1']['effective_rank'] for r in results]
    theoretical_rank = [r['corollary1']['theoretical_rank'] for r in results]
    
    axes[1, 0].plot(d_I_values, ranks, 'o-', label='Empirical')
    axes[1, 0].plot(d_I_values, theoretical_rank, '--', label='Theoretical (O(1/d_I))')
    axes[1, 0].set_xlabel('Intrinsic dimension (d_I)')
    axes[1, 0].set_ylabel('Effective rank')
    axes[1, 0].set_title('Corollary 1: GN rank scaling')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    
    # Diagonal norm vs d_I
    diag_norms = [np.mean(r['lemma3']['diagonal_norms']) for r in results]
    theoretical_diag = [r['lemma3']['theoretical_scaling'] for r in results]
    
    axes[1, 1].plot(d_I_values, diag_norms, 'o-', label='Empirical')
    axes[1, 1].plot(d_I_values, theoretical_diag, '--', label='Theoretical (O(d_I))')
    axes[1, 1].set_xlabel('Intrinsic dimension (d_I)')
    axes[1, 1].set_ylabel('Diagonal norm')
    axes[1, 1].set_title('Lemma 3: Diagonal norm scaling')
    axes[1, 1].legend()
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('dI_sweep_results.png', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Empirical verification of near-orthogonality theorem')
    parser.add_argument('--sweep', type=str, choices=['width', 'depth', 'dI', 'all'], default='all',
                        help='Which parameter sweep to run')
    parser.add_argument('--single', action='store_true', help='Run single experiment')
    parser.add_argument('--L', type=int, default=10, help='Number of layers')
    parser.add_argument('--W', type=int, default=128, help='Width')
    parser.add_argument('--d_I', type=int, default=10, help='Intrinsic dimension')
    
    args = parser.parse_args()
    
    if args.single:
        print(f"Running single experiment: L={args.L}, W={args.W}, d_I={args.d_I}")
        result = run_single_experiment(args.L, args.W, 100, args.d_I)
        print("\nResults:")
        print(f"Lemma 1 (Pushforward spectrum): c_estimate={result['lemma1']['c_estimate']:.4f}")
        print(f"Lemma 2 (Zero expectation): mean_cross={result['lemma2']['mean_cross_term']:.6f}")
        print(f"Lemma 3 (Diagonal norm): scaling_ratio={result['lemma3']['scaling_ratio']:.4f}")
        print(f"Theorem 1 (Cross-term bound): max_cross={result['theorem1']['max_cross_term']:.6f}, bound={result['theorem1']['theoretical_bound']:.6f}")
        print(f"Corollary 1 (GN rank): r_eff={result['corollary1']['effective_rank']:.2f}, theoretical={result['corollary1']['theoretical_rank']:.2f}")
        print(f"Cosine similarity: mean={result['cosine_similarity']['mean_cosine_similarity']:.6f}")
    else:
        if args.sweep in ['width', 'all']:
            run_width_sweep()
        if args.sweep in ['depth', 'all']:
            run_depth_sweep()
        if args.sweep in ['dI', 'all']:
            run_dI_sweep()
        
        print("Experiments completed. Results saved as PNG files.")


if __name__ == '__main__':
    main()
