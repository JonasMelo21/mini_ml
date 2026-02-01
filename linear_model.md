# Linear Model

Implements linear regression and related models.

## Theory & Math
Linear regression fits a line to data by minimizing the mean squared error:
$$
\min_w \sum_i (y_i - w^T x_i)^2
$$

## Usage
```python
from src.linear_model import least_square_fit, predict, gradient_descent_fit

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]
alpha, beta = least_square_fit(x, y)
pred = predict(alpha, beta, 6)
print(pred)
```

## Functions
- `least_square_fit(x, y)`: Computes linear regression coefficients using least squares
- `predict(alpha, beta, x_i)`: Predicts target value using alpha, beta, and feature value
- `compute_gradients(x, y, alpha, beta)`: Computes gradients of MSE cost function
- `gradient_descent_fit(x, y, epochs=1000, learning_rate=0.01)`: Finds alpha and beta using gradient descent
- `fit_statistical(x, y)`: Computes coefficients using statistical method

See the source code for more details.
