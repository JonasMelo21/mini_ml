# Linear Algebra

Basic linear algebra utilities for ML algorithms.

## Theory & Math
Linear algebra is the foundation of most ML algorithms. This module provides matrix/vector operations, dot products, norms, etc.

## Usage
```python
from src.linear_algebra import Vector

v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])
print(v1.dot_product(v2))
print(v1.magnitude())
print(v1 + v2)
print(v1 - v2)
```

## Classes
- `Vector(components)`: A vector class with the following methods:
  - `__add__(other)`: Add two vectors
  - `__sub__(other)`: Subtract two vectors
  - `__mul__(scalar)`: Multiply vector by a scalar
  - `dot_product(other)`: Calculate dot product
  - `sum_of_squares()`: Calculate sum of squares
  - `magnitude()`: Calculate vector magnitude
  - `euclidian_distance(other)`: Calculate Euclidean distance

See the source code for more details.
