# Logistic Model

Implements logistic regression for binary classification.

## Theory & Math
Logistic regression models the probability of class membership using the sigmoid function:
$$
P(y=1|x) = \sigma(w^T x + b)
$$

## Usage
```python
from src.logistic_model import fit, predict_probability, predict_class

x = [1, 2, 3, 4, 5]
y = [0, 0, 1, 1, 1]
alpha, beta = fit(x, y, epochs=1000, learning_rate=0.1)
prob = predict_probability(alpha, beta, 3.5)
pred = predict_class(alpha, beta, 3.5)
print(pred)
```

## Functions
- `predict_probability(alpha, beta, x_i)`: Returns probability of positive class
- `predict_class(alpha, beta, x_i, threshold=0.5)`: Returns predicted class (0 or 1)
- `compute_log_gradients(x, y, alpha, beta)`: Computes gradients for logistic regression
- `fit(x, y, epochs=1000, learning_rate=0.1)`: Trains the logistic model using gradient descent

See the source code for more details.
