# Near-Orthogonality of Layer Jacobians in Deep Residual Networks

Empirical verification codebase for the theoretical paper on near-orthogonality of layer Jacobians in deep residual networks.

## Overview

This repository provides code to empirically verify the theoretical results from the paper "Near-Orthogonality of Layer Jacobians in Deep Residual Networks: A Random-Matrix Proof with Explicit Constants" by AmirHosseinRasti (2026).

The paper proves that in an L-layer residual network satisfying a depth-scaled spectral norm bound, the full-parameter Jacobian contributions from distinct layers are asymptotically near-orthogonal. This codebase implements the empirical verification of all key theoretical results.

## Theoretical Results Verified

### Lemma 1: Pushforward Spectrum
**Claim:** `σ_min(Φ) ≥ e^(-c), σ_max ≤ e^c`

**Verification:** Compute the pushforward operator `Φ_{L←l}` and measure its singular values. The bounds should hold uniformly across layers.

### Lemma 2: Zero Expectation
**Claim:** `E[tr(Z_lᵀZ_{l'})] = 0 for l ≠ l'`

**Verification:** Sample cross-terms across many random initializations and verify that the mean is statistically indistinguishable from zero.

### Lemma 3: Diagonal Norm
**Claim:** `||J_θ^(l)||_F² = Θ(e^(2c)·d_I·W/L)`

**Verification:** Measure the Frobenius norm per layer and verify scaling with width `W`, depth `L`, and intrinsic dimension `d_I`.

### Theorem 1: Cross-Term Bound (Main Result)
**Claim:** `|⟨J_θ^(l), J_θ^(l')⟩_F| ≤ C·√(d_I·W)/L` with probability ≥ 1−2exp(−d_I)

**Verification:** Measure cross-term inner products across layers and verify they scale as `O(√(d_I·W)/L)`.

### Corollary 1: Gauss-Newton Rank Scaling
**Claim:** `r_eff(G) = Ω(L·W_min/d_I)`

**Verification:** Compute the effective rank of the Gauss-Newton matrix and verify it scales multiplicatively with depth and width.

### Falsifiable Prediction (§5.2)
**Claim:** `cosine_sim(J^(l), J^(l')) ~ O(L/√(d_I·W)) → 0 as W grows`

**Verification:** Run experiments across width/depth/d_I grids and measure cosine similarity between per-layer Jacobians.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Single Experiment

```bash
python experiments.py --single --L 10 --W 128 --d_I 10
```

This runs a single experiment with:
- L = 10 layers
- W = 128 width
- d_I = 10 intrinsic dimension

### Run Parameter Sweeps

```bash
# Width sweep
python experiments.py --sweep width

# Depth sweep
python experiments.py --sweep depth

# Intrinsic dimension sweep
python experiments.py --sweep dI

# All sweeps
python experiments.py --sweep all
```

## Code Structure

```
.
├── LICENSE                 # MIT License
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── paper.tex              # LaTeX source for the paper
├── paper.pdf              # Compiled PDF of the paper
├── experiments.py         # Main experiment script
└── src/
    ├── __init__.py
    ├── resnet.py          # ResNet implementation
    ├── jacobian.py        # Jacobian computation utilities
    └── verification.py    # Verification functions for each theorem
```

## Key Components

### ResNet Implementation (`src/resnet.py`)
- `ResidualBlock`: Single residual block F_l(x; θ_l)
- `ResNet`: L-layer residual network with configurable width and depth
- He initialization (variance = 2/W)

### Jacobian Computation (`src/jacobian.py`)
- `compute_per_layer_jacobians`: Computes J_θ^(l) = Φ_{L←l} · Z_l for each layer
- `compute_pushforward_operators`: Computes Φ_{L←l} = ∏_{k=l}^{L-1} (I + J_{F_k})
- `compute_cross_term_matrix`: Computes G_{ll'} = (J_θ^(l))^T J_θ^(l')
- `compute_diagonal_norms`: Computes ||J_θ^(l)||_F² for each layer

### Verification Functions (`src/verification.py`)
- `verify_lemma1_pushforward_spectrum`: Verifies pushforward bi-Lipschitz bounds
- `verify_lemma2_zero_expectation`: Verifies zero mean of cross-terms
- `verify_lemma3_diagonal_norm`: Verifies diagonal norm scaling
- `verify_theorem1_cross_term_bound`: Verifies main cross-term bound
- `verify_corollary1_gn_rank`: Verifies Gauss-Newton rank scaling
- `verify_cosine_similarity_scaling`: Verifies falsifiable prediction from §5.2

## Expected Results

The experiments should confirm:

1. **Cross-term scaling:** Cross-terms decrease as `O(√(d_I·W)/L)` with width and depth
2. **Diagonal scaling:** Diagonal terms scale as `Θ(d_I·W/L)` with width and depth
3. **Rank scaling:** Effective rank scales as `Ω(L·W/d_I)` (multiplicative in depth and width)
4. **Cosine similarity:** Decreases as `O(L/√(d_I·W))` and approaches zero for large width

If the empirical curves match the theoretical predictions, the theorem is confirmed. If they don't, you've found a counterexample.

## Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- SciPy >= 1.10.0
- tqdm >= 4.65.0

## License

MIT License

Copyright (c) 2026 AmirHosseinRasti

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Citation

If you use this codebase, please cite the accompanying paper:

```
@article{rasti2026near,
  title={Near-Orthogonality of Layer Jacobians in Deep Residual Networks: A Random-Matrix Proof with Explicit Constants},
  author={AmirHosseinRasti},
  journal={arXiv preprint},
  year={2026}
}
```

## Contact

For questions or issues, please open an issue on the repository or contact AmirHosseinRasti.
