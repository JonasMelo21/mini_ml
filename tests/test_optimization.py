from src.linear_algebra import Vector
from src.optimization import gradient_step, sum_of_squares_gradient
import math

def test_gradient_descent_convergence():
    # 1. Começamos num ponto aleatório longe do zero (Topo da montanha)
    # O mínimo dessa função é obviamente [0, 0, 0]
    start_point = Vector([10.0, -10.0, 5.0])
    
    # 2. Definimos um learning rate (tamanho do passo)
    # Se for muito grande, ele pula o vale. Se for muito pequeno, demora muito.
    learning_rate = 0.01
    
    current_point = start_point
    
    # 3. Rodamos o loop de aprendizado (Epochs)
    for _ in range(1000): # Damos 1000 passos
        # Calculamos a inclinação onde estamos
        grad = sum_of_squares_gradient(current_point)
        
        # Damos um passo para baixo
        current_point = gradient_step(current_point, grad, learning_rate)
    
    # 4. Verificamos se chegamos perto de zero
    # A magnitude deve ser muuuito pequena (próxima de 0.0)
    assert math.isclose(current_point.magnitude(), 0.0, abs_tol=1e-5)
    
    # Check individual components
    assert math.isclose(current_point.components[0], 0.0, abs_tol=1e-5)