import math
from src.linear_algebra import Vector

def test_vector_addition():
    v1 = Vector([1, 2])
    v2 = Vector([3, 4])
    result = v1 + v2
    # The assert checks if the expression is True. If not, it raises an error.
    assert result.components == [4, 6]

def test_vector_subtraction():
    v1 = Vector([1,2,3])
    v2 = Vector([4,5,6])
    result = v1 - v2
    assert result.components == [-3,-3,-3]

def test_dot_product():
    v1 = Vector([1, 2])
    v2 = Vector([3, 4])
    # 1*3 + 2*4 = 3 + 8 = 11
    assert v1.dot_product(v2) == 11

def test_magnitude():
    # Vector [3, 4] should have magnitude 5 (3-4-5 triangle)
    v = Vector([3, 4])
    assert v.magnitude() == 5.0

def test_magnitude_float():
    v = Vector([1, 1])
    # Square root of 2 is approx 1.414...
    # For floats, use math.isclose because computers have decimal errors
    assert math.isclose(v.magnitude(), 1.41421356, rel_tol=1e-5)

def test_distance():
    v1 = Vector([1, 1])
    v2 = Vector([4, 5])
    # The difference is (4-1, 5-1) = (3, 4)
    # The magnitude of (3, 4) is 5.
    dist = v1.euclidian_distance(v2)
    assert dist == 5.0