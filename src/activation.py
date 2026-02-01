from math import exp 

def sigmoid(z: float) -> float:
    """
    Computes the sigmoid function: 1 / (1 + e^-z).
    Args:
        z (float): The input value.
    Returns:
        float: A value between 0 and 1 (probability).
    """
    return 1 / (1 + exp(-z))
