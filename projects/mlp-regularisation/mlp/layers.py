"""
layers.py
Neural network layer implementations from scratch using NumPy.

Includes: Linear, ReLU, Softmax, Dropout
"""

import numpy as np


class Layer:
    """Base class for all layers."""

    def fprop(self, inputs, **kwargs):
        raise NotImplementedError

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        raise NotImplementedError


class LinearLayer(Layer):
    """Fully connected linear layer with optional weight penalty."""

    def __init__(self, input_dim: int, output_dim: int, weights_init_std: float = 0.1):
        self.weights = np.random.normal(0, weights_init_std, (input_dim, output_dim))
        self.biases  = np.zeros(output_dim)

    def fprop(self, inputs, **kwargs):
        return inputs @ self.weights + self.biases

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        grads_wrt_inputs   = grads_wrt_outputs @ self.weights.T
        self.grad_weights  = inputs.T @ grads_wrt_outputs
        self.grad_biases   = grads_wrt_outputs.sum(axis=0)
        return grads_wrt_inputs

    @property
    def params(self):
        return [self.weights, self.biases]

    @property
    def grads(self):
        return [self.grad_weights, self.grad_biases]


class ReluLayer(Layer):
    """Rectified Linear Unit activation."""

    def fprop(self, inputs, **kwargs):
        return np.maximum(0, inputs)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        return grads_wrt_outputs * (inputs > 0)


class SoftmaxLayer(Layer):
    """Softmax activation for multi-class output."""

    def fprop(self, inputs, **kwargs):
        exp = np.exp(inputs - inputs.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def bprop(self, inputs, outputs, grads_wrt_outputs):
        # Combined with cross-entropy loss the gradient simplifies
        return outputs - grads_wrt_outputs


class DropoutLayer(Layer):
    """
    Dropout regularisation layer using inverted dropout.

    During training, each neuron is independently zeroed with probability
    `drop_prob`. Remaining activations are scaled by 1/(1-drop_prob) so
    the expected output magnitude is unchanged at inference.

    Args:
        drop_prob: Probability of zeroing a neuron. Default 0.5.
    """

    def __init__(self, drop_prob: float = 0.5):
        assert 0.0 <= drop_prob < 1.0, "drop_prob must be in [0, 1)"
        self.drop_prob = drop_prob
        self.mask: np.ndarray | None = None

    def fprop(self, inputs: np.ndarray, stochastic: bool = True, **kwargs) -> np.ndarray:
        if stochastic:
            self.mask = (np.random.uniform(size=inputs.shape) > self.drop_prob).astype(float)
            return inputs * self.mask / (1.0 - self.drop_prob)
        # Inference: return inputs unchanged (scaling already done in training)
        return inputs

    def bprop(self, inputs: np.ndarray, outputs: np.ndarray,
              grads_wrt_outputs: np.ndarray) -> np.ndarray:
        assert self.mask is not None, "fprop must be called before bprop"
        return grads_wrt_outputs * self.mask / (1.0 - self.drop_prob)
