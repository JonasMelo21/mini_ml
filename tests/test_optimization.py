from src.linear_algebra import Vector
from src.optimization import gradient_step, sum_of_squares_gradient
import math

def test_gradient_descent_convergence():
    # 1. Start at a random point far from zero (top of the mountain)
    # The minimum of this function is obviously [0, 0, 0]
    start_point = Vector([10.0, -10.0, 5.0])
    # 2. Set a learning rate (step size)
    # If too large, it jumps over the valley. If too small, it takes too long.
    learning_rate = 0.01
    current_point = start_point
    # 3. Run the learning loop (epochs)
    for _ in range(1000): # Take 1000 steps
        # Calculate the slope at the current point
        grad = sum_of_squares_gradient(current_point)
        # Take a step down
        current_point = gradient_step(current_point, grad, learning_rate)
    # 4. Check if we reached near zero
    # The magnitude should be very small (close to 0.0)
    assert math.isclose(current_point.magnitude(), 0.0, abs_tol=1e-5)
    # Check individual components
    assert math.isclose(current_point.components[0], 0.0, abs_tol=1e-5)