# Linear Algebra

Core linear algebra utilities for ML algorithms, built around the `Vector` class.

## Theory & Math
Linear algebra is the foundation of most ML algorithms. This module provides vector operations: addition, subtraction, scalar multiplication, dot products, norms, distances, and mean calculations.

## Usage
```python
from src.linear_algebra import Vector, vector_mean, euclidian_distance

# Create vectors
v1 = Vector([1, 2, 3])
v2 = Vector([4, 5, 6])

# Vector operations
v3 = v1 + v2                          # Addition
v4 = v1 - v2                          # Subtraction
v5 = v1 * 2                           # Scalar multiplication

# Vector properties
dot_prod = v1.dot_product(v2)        # Dot product
magnitude = v1.magnitude()            # Magnitude/Length
distance = v1.euclidian_distance(v2) # Euclidean distance

# Utility functions
mean_vector = vector_mean([v1, v2])   # Mean of vectors
dist = euclidian_distance(v1, v2)    # Distance between two vectors
```

## Classes

### Vector
Represents a mathematical vector with methods for common operations.

**Constructor:**
- `Vector(components)`: Create a vector from a list of numbers

**Methods:**
- `__add__(other)`: Add two vectors → `v1 + v2`
- `__sub__(other)`: Subtract vectors → `v1 - v2`
- `__mul__(scalar)`: Scalar multiplication → `v * 2`
- `dot_product(other)`: Dot product between two vectors
- `sum_of_squares()`: Sum of squared components
- `magnitude()`: Euclidean norm (length) of the vector
- `euclidian_distance(other)`: Distance to another vector

## Functions
- `vector_mean(vectors)`: Calculate the centroid (mean) of multiple vectors
- `euclidian_distance(v1, v2)`: Calculate Euclidean distance between two vectors

See the source code for more details.

