from src.statistics import mean, median, variance, std_deviation,covariance,correlation
from math import isclose 
import pytest

def test_mean():
    assert mean([1, 2, 3]) == 2.0
    assert mean([0, 0, 4, 4]) == 2.0

def test_median_odd():
    # Odd list: [1, 10, 2, 9, 5] -> sorted: [1, 2, 5, 9, 10] -> middle: 5
    assert median([1, 10, 2, 9, 5]) == 5

def test_median_even():
    # Even list: [1, 9, 2, 10] -> sorted: [1, 2, 9, 10] -> middle: (2+9)/2 = 5.5
    assert median([1, 9, 2, 10]) == 5.5

def test_variance():
    # Data: [0, 0, 4, 4] -> Mean: 2
    # Deviations: [-2, -2, 2, 2]
    # Squares: [4, 4, 4, 4] -> Sum: 16
    # Division (n-1): 16 / 3 = 5.333...
    data = [0, 0, 4, 4]
    assert isclose(variance(data), 5.333333, rel_tol=1e-5)

def test_std_dev():
    data = [1, 2, 3]
    # Variance is 1.0. Square root of 1 is 1.
    assert std_deviation(data) == 1.0

def test_covariance():
    xs = [1, 2, 3]
    ys = [1, 2, 3]
    # Means: 2 and 2.
    # Deviations: [-1, 0, 1] and [-1, 0, 1]
    # Dot: (-1*-1) + (0*0) + (1*1) = 1 + 0 + 1 = 2
    # Cov: 2 / (3-1) = 1.0
    assert covariance(xs, ys) == 1.0

def test_correlation_perfect():
    xs = [1, 2, 3]
    ys = [2, 4, 6] # Y is exactly 2*X
    # Should have perfect correlation 1.0
    assert correlation(xs, ys) == 1.0

def test_correlation_inverse():
    xs = [1, 2, 3]
    ys = [3, 2, 1] # Y decreases when X increases
    assert correlation(xs, ys) == -1.0

def test_correlation_zero():
    # Sem relação clara
    xs = [1, 2, 3]
    ys = [10, 10, 10] # Y não muda nada
    # Desvio padrão de Y é 0. Divisão por zero deve ser tratada.
    assert correlation(xs, ys) == 0.0