# Activation Functions

This module implements common activation functions used in neural networks.

## Theory & Math
Activation functions introduce non-linearity into neural networks. Common examples:
- **Sigmoid**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- **ReLU**: $f(x) = \max(0, x)$
- **Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

## Usage
```python
from src.activation import sigmoid, relu, tanh

print(sigmoid(0.5))
print(relu([-1, 0, 2]))
```

## Functions
- `sigmoid(x)`
- `relu(x)`
- `tanh(x)`

See the source code for more details.
