# Logistic Model

Implements logistic regression for binary classification.

## Theory & Math
Logistic regression models the probability of class membership using the sigmoid function:
$$
P(y=1|x) = \sigma(w^T x + b)
$$

## Usage
```python
from src.logistic_model import LogisticRegression

X = [[1, 2], [2, 3], [3, 4]]
y = [0, 1, 1]
model = LogisticRegression()
model.fit(X, y)
pred = model.predict([[4, 5]])
print(pred)
```

## Classes
- `LogisticRegression()`

See the source code for more details.
