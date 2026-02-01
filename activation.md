# Activation Functions

This module implements common activation functions used in neural networks.

## Theory & Math
Activation functions introduce non-linearity into neural networks. This module currently implements:
- **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$

## Usage
```python
from src.activation import sigmoid

print(sigmoid(0.5))
```

## Functions
- `sigmoid(z)`: Computes the sigmoid function, returns value between 0 and 1

See the source code for more details.
