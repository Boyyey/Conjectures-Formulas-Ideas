# Invariant Manifold Transform (IMT)

A differential-geometric framework for causal representation learning that resolves the fundamental observational equivalence between causal and confounded systems without randomized trials or pre-specified causal graphs.

## Overview

The Invariant Manifold Transform (IMT) treats environmental shifts as continuous transformation groups acting on the parameter space of a Riemannian statistical manifold. It formalizes the **Environmental Lie Algebra** and proves that the structural causal mechanism is exactly the maximal invariant set.

### Key Features

- **Geodesic Projection Algorithm**: Enforces invariance constraints exactly via projection rather than approximate penalty methods
- **Two Projection Variants**:
  - Conservative null-space projection (when environmental velocities are not directly accessible)
  - Exact horizontal projection (when environmental velocities are estimable)
- **Spectral Gap Theorem**: Characterizes the dimension–identifiability tradeoff
- **Bridge Theorem**: Establishes the connection between causal subspaces and environmental score covariances
- **Consistency Guarantees**: Convergence proofs with geometric rates under standard conditions

## Installation

```bash
git clone https://github.com/yourusername/IMT.git
cd IMT
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0

## Usage

### Basic Example

```python
import numpy as np
from src.imt import IMT, IMTConfig

# Define your model (log-likelihood gradient)
def model(X, y, theta):
    # Compute gradient of log-likelihood w.r.t. theta
    # Returns: gradient vector of shape (d,)
    pass

# Define loss function
def loss_fn(theta, environments):
    # Compute loss and gradient
    # Returns: (loss, gradient)
    pass

# Prepare environment data
environments = [
    (X_e1, y_e1),  # Environment 1
    (X_e2, y_e2),  # Environment 2
    # ... more environments
]

# Initialize IMT
config = IMTConfig(
    learning_rate=0.01,
    max_iterations=1000,
    projection_variant="conservative"
)
imt = IMT(config)

# Fit the model
theta_init = np.random.randn(d)
theta_learned = imt.fit(
    model=model,
    loss_fn=loss_fn,
    environments=environments,
    theta_init=theta_init
)

# Get diagnostics
diagnostics = imt.diagnostics()
print(f"Spectral gap: {diagnostics['spectral_gap']}")
print(f"Rank deficiency: {diagnostics['rank_deficiency']}")
```

### Projection Variants

#### Conservative Projection
Use when environmental velocity basis is not directly accessible:

```python
config = IMTConfig(projection_variant="conservative")
imt = IMT(config)
```

This projects gradients onto the null space of the aggregate score matrix using the Moore-Penrose pseudoinverse.

#### Exact Horizontal Projection
Use when environmental velocities are estimable (via finite differences or PCA):

```python
config = IMTConfig(projection_variant="exact")
imt = IMT(config)
```

This enforces the exact Fisher-metric-orthogonal projection onto the horizontal subspace.

### Computing Causal Subspace

```python
# Compute environmental score matrices
score_matrices = imt.compute_environmental_score_matrices(
    model, environments, theta
)

# Compute causal subspace basis
causal_basis = imt.compute_causal_subspace(
    score_matrices, 
    method="null_space"
)
```

### Checking Spanning Condition

The Spanning Condition ensures that environmental score images span the full vertical subspace:

```python
vertical_basis = imt.estimate_vertical_basis(score_matrices)
condition_holds, delta = imt.check_spanning_condition(
    score_matrices, vertical_basis
)

if condition_holds:
    print("Spanning Condition satisfied")
else:
    print(f"Spanning Condition failed (delta={delta})")
```

## Algorithm Details

### Geodesic Projection Algorithm

The algorithm enforces the geometric constraint that gradients must lie in the causal (horizontal) subspace at each iteration:

**Conservative variant:**
```
θ_{t+1} = θ_t - η_t · Π_{ker(Ŝ(θ_t))} · ∇_θ L(θ_t)
```

**Exact horizontal variant:**
```
θ_{t+1} = θ_t - η_t · Π_H(θ_t) · ∇_θ L(θ_t)
```

where Π projects onto the causal subspace.

### Key Theorems

1. **Spectral Gap Theorem**: Characterizes when the causal core is identifiable and when it is destroyed by transitive environmental action
2. **Bridge Theorem**: Establishes the two-part relationship between causal subspaces and environmental score null spaces
3. **Structural Stability Theorem**: Quantifies how strongly spurious proxies are expelled from the causal null space
4. **Consistency Theorem**: Provides convergence guarantees with geometric rates

## Citation

```bibtex
@article{rasti2026imt,
  title={The Invariant Manifold Transform: Causality as Symmetry, Foliations, and the Geometry of Causal Identifiability},
  author={Rasti, Amir Hossein},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License

## Reference

See `Paper/paper.tex` for the complete mathematical formulation and proofs.
