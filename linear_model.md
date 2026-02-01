# Linear Model

Implements linear regression and related models.

## Theory & Math
Linear regression fits a line to data by minimizing the mean squared error:
$$
\min_w \sum_i (y_i - w^T x_i)^2
$$

## Usage
```python
from src.linear_model import LinearRegression

X = [[1, 2], [2, 3], [3, 4]]
y = [3, 5, 7]
model = LinearRegression()
model.fit(X, y)
pred = model.predict([[4, 5]])
print(pred)
```

## Classes
- `LinearRegression()`

See the source code for more details.
