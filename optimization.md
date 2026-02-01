# Optimization

Optimization algorithms for training ML models.

## Theory & Math
Optimization finds the best parameters for a model by minimizing a loss function. Common algorithms:
- **Gradient Descent**: $w \leftarrow w - \eta \nabla L(w)$

## Usage
```python
from src.optimization import gradient_step, sum_of_squares_gradient
from src.linear_algebra import Vector

v = Vector([3.0, 4.0])
gradient = sum_of_squares_gradient(v)
new_v = gradient_step(v, gradient, step_size=0.01)
print(new_v)
```

## Functions
- `gradient_step(v, gradient, step_size)`: Moves vector v in the opposite direction of the gradient
- `sum_of_squares_gradient(v)`: Calculates gradient of f(v) = sum(v_i ^ 2)

See the source code for more details.
