from src.statistics import mean, median, variance, std_deviation,covariance,correlation
from math import isclose 
import pytest

def test_mean():
    assert mean([1, 2, 3]) == 2.0
    assert mean([0, 0, 4, 4]) == 2.0

def test_median_odd():
    # Lista ímpar: [1, 10, 2, 9, 5] -> ordenado: [1, 2, 5, 9, 10] -> meio: 5
    assert median([1, 10, 2, 9, 5]) == 5

def test_median_even():
    # Lista par: [1, 9, 2, 10] -> ordenado: [1, 2, 9, 10] -> meio: (2+9)/2 = 5.5
    assert median([1, 9, 2, 10]) == 5.5

def test_variance():
    # Dados: [0, 0, 4, 4] -> Média: 2
    # Desvios: [-2, -2, 2, 2]
    # Quadrados: [4, 4, 4, 4] -> Soma: 16
    # Divisão (n-1): 16 / 3 = 5.333...
    data = [0, 0, 4, 4]
    assert isclose(variance(data), 5.333333, rel_tol=1e-5)

def test_std_dev():
    data = [1, 2, 3]
    # Variância é 1.0. Raiz de 1 é 1.
    assert std_deviation(data) == 1.0

def test_covariance():
    xs = [1, 2, 3]
    ys = [1, 2, 3]
    # Médias: 2 e 2.
    # Desvios: [-1, 0, 1] e [-1, 0, 1]
    # Dot: (-1*-1) + (0*0) + (1*1) = 1 + 0 + 1 = 2
    # Cov: 2 / (3-1) = 1.0
    assert covariance(xs, ys) == 1.0

def test_correlation_perfect():
    xs = [1, 2, 3]
    ys = [2, 4, 6] # Y é exatamente 2*X
    # Devem ter correlação perfeita 1.0
    assert correlation(xs, ys) == 1.0

def test_correlation_inverse():
    xs = [1, 2, 3]
    ys = [3, 2, 1] # Y diminui quando X aumenta
    assert correlation(xs, ys) == -1.0

def test_correlation_zero():
    # Sem relação clara
    xs = [1, 2, 3]
    ys = [10, 10, 10] # Y não muda nada
    # Desvio padrão de Y é 0. Divisão por zero deve ser tratada.
    assert correlation(xs, ys) == 0.0