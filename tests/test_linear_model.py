from src.linear_model import least_square_fit, predict
import pytest

def test_perfect_line():
    # Caso perfeito: y = 10 + 2x
    # Se x=1, y=12. Se x=2, y=14.
    x = [1, 2, 3]
    y = [12, 14, 16]
    
    alpha, beta = least_square_fit(x, y)
    
    assert beta == 2.0
    assert alpha == 10.0

def test_prediction():
    alpha = 10.0
    beta = 2.0
    # Se tenho 5 anos de xp, quanto ganho? 10 + 2*5 = 20
    assert predict(alpha, beta, x_i=5) == 20.0