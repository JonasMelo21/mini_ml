import math
from src.linear_algebra import Vector

def test_vector_addition():
    v1 = Vector([1, 2])
    v2 = Vector([3, 4])
    result = v1 + v2
    # O assert verifica se a expressão é True. Se não for, ele grita erro.
    assert result.components == [4, 6]

def test_dot_product():
    v1 = Vector([1, 2])
    v2 = Vector([3, 4])
    # 1*3 + 2*4 = 3 + 8 = 11
    assert v1.dot_product(v2) == 11

def test_magnitude():
    # Vetor 3, 4 deve ter magnitude 5 (Triângulo 3-4-5)
    v = Vector([3, 4])
    assert v.magnitude() == 5.0

def test_magnitude_float():
    v = Vector([1, 1])
    # Raiz de 2 é aprox 1.414...
    # Para comparar floats, usamos math.isclose pq computador erra decimal
    assert math.isclose(v.magnitude(), 1.41421356, rel_tol=1e-5)