import math
from src.linear_algebra import Vector
from src.preprocessing import StandardScaler
from src.statistics import mean, std_deviation

def test_standard_scaler_basic():
    """
    Tests if the scaler centers the data at 0 and scales to unit variance.
    """
    # Arrange: Simple data
    # Column 0: [0, 10] -> Mean 5
    # Column 1: [0, 10] -> Mean 5
    data = [Vector([0.0, 0.0]), Vector([10.0, 10.0])]
    # Act: Fit and Transform
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # Assert 1: Values changed?
    # Value 0 is below the mean, so should be negative
    assert scaled_data[0].components[0] < 0
    # Value 10 is above the mean, so should be positive
    assert scaled_data[1].components[0] > 0
    # Assert 2: Statistical properties (the real proof)
    # Extract the first column of the transformed data
    transformed_matrix = [v.components for v in scaled_data]
    col0 = list(zip(*transformed_matrix))[0]  # Take all items from column 0
    # The mean should be 0 (or very close, like 0.00000001)
    assert math.isclose(mean(col0), 0.0, abs_tol=1e-5)
    # The standard deviation should be 1
    assert math.isclose(std_deviation(col0), 1.0, abs_tol=1e-5)

def test_standard_scaler_constant_column():
    """
    Tests the edge case: a column where all values are the same.
    The standard deviation is 0. The scaler should avoid division by zero.
    """
    # Column 0 varies, Column 1 is constant (5, 5)
    data = [Vector([0.0, 5.0]), Vector([10.0, 5.0])]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # Column 1 (index 1) should become 0 (since value - mean = 5 - 5 = 0)
    # And should not raise ZeroDivisionError
    assert scaled_data[0].components[1] == 0.0
    assert scaled_data[1].components[1] == 0.0