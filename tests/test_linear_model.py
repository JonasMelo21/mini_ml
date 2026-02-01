from src.linear_model import least_square_fit, predict,gradient_descent_fit,mean,variance,covariance, fit_statistical
import math

import pytest

def test_perfect_line():
    # Perfect case: y = 10 + 2x
    # If x=1, y=12. If x=2, y=14.
    x = [1, 2, 3]
    y = [12, 14, 16]
    alpha, beta = least_square_fit(x, y)
    assert beta == 2.0
    assert alpha == 10.0

def test_prediction():
    alpha = 10.0
    beta = 2.0
    # If I have 5 years of experience, what do I earn? 10 + 2*5 = 20
    assert predict(alpha, beta, x_i=5) == 20.0


def test_gradient_descent_convergence():
    # Simple data: y = 2x + 1
    x = [1, 2, 3, 4, 5]
    y = [3, 5, 7, 9, 11]
    # Train with GD
    alpha, beta = gradient_descent_fit(x, y, epochs=2000, learning_rate=0.01)
    # Check if it gets close to alpha=1.0 and beta=2.0
    # GD is never exact, so we use error margin (abs_tol)
    assert math.isclose(alpha, 1.0, abs_tol=0.1)
    assert math.isclose(beta, 2.0, abs_tol=0.1)

def test_fit_statistical():
    # Test data: y = 5x + 10
    x = [1,2,3,4]
    y = [15,20,25,30]
    alpha, beta = fit_statistical(x, y)
    assert math.isclose(alpha, 5.0, abs_tol=0.1)
    assert math.isclose(beta, 10.0, abs_tol=0.1)
    # Second test data: 1x + 0
    x = [1,2,3]
    y = [1,2,3]

    new_alpha,new_beta = fit_statistical(x,y)

    assert math.isclose(new_alpha,1.0,abs_tol=0.1)
    assert math.isclose(new_beta,0,abs_tol=0.1)