"""
Invariant Manifold Transform (IMT) Implementation

This module implements the Geodesic Projection Algorithm for causal representation
learning based on the Invariant Manifold Transform framework.

Reference: "The Invariant Manifold Transform: Causality as Symmetry, Foliations,
and the Geometry of Causal Identifiability" by Amir Hossein Rasti (2026)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass


@dataclass
class IMTConfig:
    """Configuration for IMT algorithm."""
    learning_rate: float = 0.01
    max_iterations: int = 1000
    tolerance: float = 1e-6
    projection_variant: str = "conservative"  # "conservative" or "exact"
    spectral_gap_threshold: float = 1e-4
    regularization: float = 1e-8


class IMT:
    """
    Invariant Manifold Transform (IMT) Algorithm
    
    Implements the Geodesic Projection Algorithm for learning causal
    representations from non-stationary environments.
    """
    
    def __init__(self, config: Optional[IMTConfig] = None):
        """
        Initialize IMT algorithm.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or IMTConfig()
        self.score_matrices_: Optional[List[np.ndarray]] = None
        self.aggregate_score_: Optional[np.ndarray] = None
        self.spectral_gap_: Optional[float] = None
        self.rank_deficiency_: Optional[int] = None
        
    def compute_score_covariance(
        self,
        model: Callable,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute environmental score covariance matrix.
        
        Args:
            model: Function that computes log-likelihood gradient given theta
            X: Input features
            y: Target values
            theta: Current parameter vector
            
        Returns:
            Score covariance matrix S_e(theta)
        """
        # Compute score (gradient of log-likelihood)
        scores = []
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            score = model(X[i:i+1], y[i:i+1], theta)
            scores.append(score)
        
        scores = np.array(scores)  # Shape: (n_samples, d)
        
        # Compute covariance
        S_e = np.dot(scores.T, scores) / n_samples
        
        return S_e
    
    def compute_environmental_score_matrices(
        self,
        model: Callable,
        environments: List[Tuple[np.ndarray, np.ndarray]],
        theta: np.ndarray
    ) -> List[np.ndarray]:
        """
        Compute score covariance matrices for all environments.
        
        Args:
            model: Function that computes log-likelihood gradient
            environments: List of (X, y) tuples for each environment
            theta: Current parameter vector
            
        Returns:
            List of score covariance matrices, one per environment
        """
        score_matrices = []
        
        for X_e, y_e in environments:
            S_e = self.compute_score_covariance(model, X_e, y_e, theta)
            score_matrices.append(S_e)
        
        self.score_matrices_ = score_matrices
        return score_matrices
    
    def compute_aggregate_score(self, score_matrices: List[np.ndarray]) -> np.ndarray:
        """
        Compute aggregate score matrix across all environments.
        
        Args:
            score_matrices: List of score covariance matrices
            
        Returns:
            Aggregate score matrix S_agg = sum_e S_e
        """
        S_agg = np.sum(score_matrices, axis=0)
        self.aggregate_score_ = S_agg
        return S_agg
    
    def conservative_projection(
        self,
        aggregate_score: np.ndarray,
        gradient: np.ndarray
    ) -> np.ndarray:
        """
        Conservative null-space projection (Equation 7 in paper).
        
        Projects gradient onto the null space of aggregate score matrix
        using Moore-Penrose pseudoinverse.
        
        Args:
            aggregate_score: Aggregate score matrix S_agg
            gradient: Gradient of loss function
            
        Returns:
            Projected gradient
        """
        # Compute pseudoinverse
        S_pinv = np.linalg.pinv(aggregate_score, rcond=self.config.regularization)
        
        # Projection matrix onto null space
        Pi_ker = np.eye(aggregate_score.shape[0]) - np.dot(S_pinv, aggregate_score)
        
        # Project gradient
        projected_gradient = np.dot(Pi_ker, gradient)
        
        return projected_gradient
    
    def exact_horizontal_projection(
        self,
        score_matrices: List[np.ndarray],
        vertical_basis: np.ndarray,
        gradient: np.ndarray,
        fisher_metric: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Exact horizontal projection (Equation 9 in paper).
        
        Projects gradient onto the horizontal subspace orthogonal to
        vertical directions under the Fisher metric.
        
        Args:
            score_matrices: List of score covariance matrices
            vertical_basis: Orthonormal basis for vertical subspace V_theta
            gradient: Gradient of loss function
            fisher_metric: Fisher information matrix (if None, uses identity)
            
        Returns:
            Projected gradient
        """
        if fisher_metric is None:
            fisher_metric = np.eye(gradient.shape[0])
        
        # Compute aggregate score
        S_agg = self.compute_aggregate_score(score_matrices)
        
        # Compute projection matrix (Equation 9)
        P = vertical_basis
        M = np.dot(P.T, np.dot(S_agg, P))
        M_inv = np.linalg.inv(M + self.config.regularization * np.eye(M.shape[0]))
        
        Pi_H = np.eye(gradient.shape[0]) - np.dot(P, np.dot(M_inv, np.dot(P.T, S_agg)))
        
        # Project gradient
        projected_gradient = np.dot(Pi_H, gradient)
        
        return projected_gradient
    
    def estimate_vertical_basis(
        self,
        score_matrices: List[np.ndarray],
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Estimate vertical subspace basis from score matrices using PCA.
        
        Args:
            score_matrices: List of score covariance matrices
            n_components: Number of vertical components. If None, estimated from gap.
            
        Returns:
            Orthonormal basis for vertical subspace
        """
        S_agg = self.compute_aggregate_score(score_matrices)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(S_agg)
        
        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Detect spectral gap
        if n_components is None:
            gaps = np.diff(eigenvalues)
            n_components = np.argmax(gaps) + 1 if len(gaps) > 0 else 0
        
        # Store spectral gap for diagnostics
        if n_components < len(eigenvalues):
            self.spectral_gap_ = eigenvalues[n_components] - eigenvalues[n_components - 1]
        else:
            self.spectral_gap_ = 0.0
        
        self.rank_deficiency_ = n_components
        
        # Vertical basis corresponds to smallest eigenvalues
        vertical_basis = eigenvectors[:, :n_components]
        
        return vertical_basis
    
    def check_spanning_condition(
        self,
        score_matrices: List[np.ndarray],
        vertical_basis: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if Spanning Condition holds (Definition 6 in paper).
        
        Args:
            score_matrices: List of score covariance matrices
            vertical_basis: Orthonormal basis for vertical subspace
            
        Returns:
            Tuple of (condition_holds, delta_value)
        """
        S_agg = self.compute_aggregate_score(score_matrices)
        
        # Project aggregate score onto vertical subspace
        Pi_V = np.dot(vertical_basis, vertical_basis.T)
        S_V = np.dot(Pi_V, np.dot(S_agg, Pi_V))
        
        # Compute minimum eigenvalue (delta)
        eigenvalues = np.linalg.eigvalsh(S_V)
        delta = np.min(eigenvalues)
        
        condition_holds = delta > self.config.spectral_gap_threshold
        
        return condition_holds, delta
    
    def fit(
        self,
        model: Callable,
        loss_fn: Callable,
        environments: List[Tuple[np.ndarray, np.ndarray]],
        theta_init: np.ndarray,
        fisher_metric: Optional[Callable] = None,
        callback: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Run Geodesic Projection Algorithm to learn causal parameters.
        
        Args:
            model: Function that computes log-likelihood gradient
            loss_fn: Function that computes loss and gradient given theta
            environments: List of (X, y) tuples for each environment
            theta_init: Initial parameter vector
            fisher_metric: Optional function to compute Fisher metric
            callback: Optional callback function for monitoring
            
        Returns:
            Learned parameter vector
        """
        theta = theta_init.copy()
        prev_loss = np.inf
        
        for iteration in range(self.config.max_iterations):
            # Compute loss and gradient
            loss, gradient = loss_fn(theta, environments)
            
            # Check convergence
            if abs(prev_loss - loss) < self.config.tolerance:
                break
            
            prev_loss = loss
            
            # Compute environmental score matrices
            score_matrices = self.compute_environmental_score_matrices(
                model, environments, theta
            )
            
            # Project gradient based on variant
            if self.config.projection_variant == "conservative":
                S_agg = self.compute_aggregate_score(score_matrices)
                projected_gradient = self.conservative_projection(S_agg, gradient)
            else:  # exact
                vertical_basis = self.estimate_vertical_basis(score_matrices)
                if fisher_metric is not None:
                    F = fisher_metric(theta, environments)
                else:
                    F = None
                projected_gradient = self.exact_horizontal_projection(
                    score_matrices, vertical_basis, gradient, F
                )
            
            # Update parameters
            theta -= self.config.learning_rate * projected_gradient
            
            # Callback
            if callback is not None:
                callback(iteration, theta, loss, projected_gradient)
        
        return theta
    
    def compute_causal_subspace(
        self,
        score_matrices: List[np.ndarray],
        method: str = "null_space"
    ) -> np.ndarray:
        """
        Compute causal subspace using Bridge Theorem (Theorem 5 in paper).
        
        Args:
            score_matrices: List of score covariance matrices
            method: Method to use ("null_space" or "horizontal")
            
        Returns:
            Basis for causal subspace
        """
        if method == "null_space":
            # Intersection of null spaces (safe inclusion)
            S_agg = self.compute_aggregate_score(score_matrices)
            eigenvalues, eigenvectors = np.linalg.eigh(S_agg)
            
            # Null space corresponds to zero eigenvalues
            null_mask = eigenvalues < self.config.spectral_gap_threshold
            causal_basis = eigenvectors[:, null_mask]
            
        elif method == "horizontal":
            # Horizontal subspace (requires vertical basis)
            vertical_basis = self.estimate_vertical_basis(score_matrices)
            # Horizontal is orthogonal complement of vertical
            causal_basis = self._orthogonal_complement(vertical_basis)
        
        return causal_basis
    
    def _orthogonal_complement(self, basis: np.ndarray) -> np.ndarray:
        """Compute orthonormal complement of a subspace basis."""
        Q, _ = np.linalg.qr(basis)
        I = np.eye(basis.shape[0])
        complement = I - np.dot(Q, Q.T)
        eigenvalues, eigenvectors = np.linalg.eigh(complement)
        complement_basis = eigenvectors[:, eigenvalues > self.config.spectral_gap_threshold]
        return complement_basis
    
    def diagnostics(self) -> Dict:
        """
        Return diagnostic information about the last fit.
        
        Returns:
            Dictionary containing spectral gap, rank deficiency, etc.
        """
        return {
            "spectral_gap": self.spectral_gap_,
            "rank_deficiency": self.rank_deficiency_,
            "spanning_condition": self.spectral_gap_ > self.config.spectral_gap_threshold
        }
