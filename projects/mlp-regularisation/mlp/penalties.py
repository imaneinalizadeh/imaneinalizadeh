"""
penalties.py
Weight penalty implementations for regularisation.

L1  — Lasso: promotes sparse weights
L2  — Ridge: penalises large weights
L1L2 — Elastic net: combines both
"""

import numpy as np


class L1Penalty:
    """
    L1 (Lasso) regularisation penalty.
    penalty(w) = coeff * sum(|w|)
    grad(w)    = coeff * sign(w)
    """

    def __init__(self, coeff: float = 1e-4):
        self.coeff = coeff

    def __call__(self, weights: np.ndarray) -> float:
        return self.coeff * np.abs(weights).sum()

    def grad(self, weights: np.ndarray) -> np.ndarray:
        return self.coeff * np.sign(weights)


class L2Penalty:
    """
    L2 (Ridge) regularisation penalty.
    penalty(w) = 0.5 * coeff * sum(w^2)
    grad(w)    = coeff * w
    """

    def __init__(self, coeff: float = 1e-4):
        self.coeff = coeff

    def __call__(self, weights: np.ndarray) -> float:
        return 0.5 * self.coeff * (weights ** 2).sum()

    def grad(self, weights: np.ndarray) -> np.ndarray:
        return self.coeff * weights


class L1L2MixPenalty:
    """
    Elastic net regularisation penalty.
    Combines L1 and L2 with independent coefficients.

    penalty(w) = l1_coeff * sum(|w|) + 0.5 * l2_coeff * sum(w^2)
    grad(w)    = l1_coeff * sign(w) + l2_coeff * w
    """

    def __init__(self, l1_coeff: float = 1e-4, l2_coeff: float = 1e-4):
        self.l1 = L1Penalty(l1_coeff)
        self.l2 = L2Penalty(l2_coeff)

    def __call__(self, weights: np.ndarray) -> float:
        return self.l1(weights) + self.l2(weights)

    def grad(self, weights: np.ndarray) -> np.ndarray:
        return self.l1.grad(weights) + self.l2.grad(weights)
