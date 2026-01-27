from typing import List, Callable
from src.linear_algebra import Vector

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """
    Moves the vector v in the opposite direction of the gradient.

    Args:
        v (Vector): The current point (parameter vector).
        gradient (Vector): The direction of steepest ascent (derivative).
        step_size (float): The step size (learning rate).

    Returns:
        Vector: The new point after the step.
    """
    # TIP: Remember that you have already implemented scalar multiplication (__mul__)
    # and subtraction (__sub__) in the Vector class.
    # The formula is: new_v = v - (gradient * step_size)
    # But wait... does your Vector class support subtraction? Yes.
    # Does it support scalar multiplication? Yes.
    # So this should be 1 elegant line of code.
    return v - (gradient * step_size)

def sum_of_squares_gradient(v: Vector) -> Vector:
    """
    Calculates the gradient of the function f(v) = sum(v_i ^ 2).
    The derivative of x^2 is 2x.
    Therefore, the gradient of [x, y, z] is [2x, 2y, 2z].

    Args:
        v (Vector): The input vector.

    Returns:
        Vector: The gradient vector.
    """
    # Use list comprehension to generate the new components
    return Vector([2 * x for x in v.components])