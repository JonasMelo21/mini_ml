# Optimization

Optimization algorithms for training ML models.

## Theory & Math
Optimization finds the best parameters for a model by minimizing a loss function. Common algorithms:
- **Gradient Descent**: $w \leftarrow w - \eta \nabla L(w)$

## Usage
```python
from src.optimization import gradient_descent

def loss(w):
    return (w - 2) ** 2

def grad(w):
    return 2 * (w - 2)

w_opt = gradient_descent(loss, grad, w_init=0.0)
print(w_opt)
```

## Functions
- `gradient_descent(loss_fn, grad_fn, w_init, lr=0.01, n_iter=100)`

See the source code for more details.
