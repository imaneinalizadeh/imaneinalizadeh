# MLP Regularisation — Neural Network Overfitting Study

A from-scratch implementation of regularisation techniques for multi-layer perceptrons in NumPy. Implements Dropout, L1, L2, and combined L1L2 penalties, with systematic experiments demonstrating their effect on overfitting across varying network widths and depths.

---

## What It Does

Trains a configurable MLP on MNIST-style data with and without regularisation, tracking training vs validation accuracy to demonstrate overfitting. Implements all regularisation layers from scratch using NumPy without deep learning frameworks.

---

## Implemented Components

| Component | Description |
|-----------|-------------|
| `DropoutLayer` | Stochastic neuron masking during training (`fprop` + `bprop`) |
| `L1Penalty` | Lasso regularisation — promotes sparse weights |
| `L2Penalty` | Ridge regularisation — penalises large weights |
| `L1L2MixPenalty` | Elastic net — combined L1 and L2 penalty |
| `ReluLayer` | Rectified linear activation |
| `SoftmaxLayer` | Probabilistic output with cross-entropy loss |

---

## Repository Structure

```
mlp-regularisation/
├── mlp/
│   ├── layers.py              # All layer implementations
│   ├── penalties.py           # L1, L2, L1L2 penalty classes
│   ├── optimisers.py          # SGD with momentum
│   ├── data_providers.py      # Dataset loading and batching
│   └── errors.py              # Cross-entropy and MSE losses
├── experiments/
│   ├── task1_width_depth.py   # Vary network width and depth
│   ├── task2_regularisation.py # Compare regularisation methods
│   └── plot_results.py        # Generate all report figures
├── report/
│   └── report.pdf             # Full coursework report
├── requirements.txt
└── README.md
```

---

## Getting Started

### Install

```bash
pip install -r requirements.txt
```

### Run experiments

```bash
# Task 1: width and depth overfitting
python experiments/task1_width_depth.py

# Task 2: regularisation comparison
python experiments/task2_regularisation.py

# Generate plots
python experiments/plot_results.py
```

---

## Key Results

| Configuration | Train Acc | Val Acc | Gap |
|--------------|-----------|---------|-----|
| Baseline (no regularisation) | 98.2% | 87.4% | 10.8% |
| + Dropout (p=0.5) | 95.1% | 91.3% | 3.8% |
| + L2 (λ=0.001) | 96.4% | 90.8% | 5.6% |
| + L1L2 Mix | 94.7% | 91.6% | 3.1% |

Dropout and elastic net produced the best generalisation on the test set.

---

## Dropout Implementation

```python
def fprop(self, inputs, stochastic=True):
    if stochastic:
        self.mask = (np.random.uniform(size=inputs.shape) > self.drop_prob)
        return inputs * self.mask / (1.0 - self.drop_prob)
    return inputs  # inference: no dropout, no scaling needed

def bprop(self, inputs, outputs, grads_wrt_outputs):
    return grads_wrt_outputs * self.mask / (1.0 - self.drop_prob)
```

Inverted dropout ensures the expected value of each neuron's output is the same at training and inference time.

---

## Dependencies

```
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.11.0
```

---

## Academic Context

**Course:** Machine Learning Practical (MLP) — MSc HPC and Data Science
**Institution:** University of Edinburgh
**Submission:** Coursework 2, November 2025
