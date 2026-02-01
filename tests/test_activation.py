import math
from src.activation import sigmoid

def test_sigmoid_zero():
    # At the center, probability should be 50%
    assert math.isclose(sigmoid(0), 0.5)

def test_sigmoid_positive_extreme():
    # Should approach 1
    assert math.isclose(sigmoid(100), 1.0, rel_tol=1e-5)

def test_sigmoid_negative_extreme():
    # Should approach 0
    assert math.isclose(sigmoid(-100), 0.0, abs_tol=1e-5)