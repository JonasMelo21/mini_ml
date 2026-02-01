from src.activation import sigmoid
from typing import List, Tuple

def predict_probability(alpha: float, beta: float, x_i: float) -> float:
    """
    Returns the probability (0 to 1) of being the positive class.
    Args:
        alpha (float): Intercept coefficient.
        beta (float): Slope coefficient.
        x_i (float): Feature value.
    Returns:
        float: Probability of positive class.
    """
    z = alpha + beta * x_i
    return sigmoid(z)

def predict_class(alpha: float, beta: float, x_i: float, threshold: float = 0.5) -> int:
    """
    Returns 1 or 0 based on the predicted probability and threshold.
    Args:
        alpha (float): Intercept coefficient.
        beta (float): Slope coefficient.
        x_i (float): Feature value.
        threshold (float): Classification threshold (default 0.5).
    Returns:
        int: Predicted class (0 or 1).
    """
    return 1 if predict_probability(alpha, beta, x_i) > threshold else 0


def compute_log_gradients(
    x: List[float],
    y: List[float],
    alpha: float,
    beta: float
) -> Tuple[float, float]:
    """
    Computes the gradients for Logistic Regression.
    The formula is similar to Linear Regression, but predictions use the sigmoid function.
    Gradient = (Prediction - Actual) * x
    Args:
        x (List[float]): Feature values.
        y (List[float]): Target values.
        alpha (float): Intercept coefficient.
        beta (float): Slope coefficient.
    Returns:
        Tuple[float, float]: (grad_alpha, grad_beta)
    """
    n = len(x)
    predictions = [predict_probability(alpha, beta, xi) for xi in x]
    errors = [pred - target for pred, target in zip(predictions, y)]
    grad_alpha = sum(errors) * (1 / n)
    grad_beta = sum([err * xi for err, xi in zip(errors, x)]) * (1 / n)
    return grad_alpha, grad_beta

def fit(
    x: List[float], 
    y: List[float], 
    epochs: int = 1000, 
    learning_rate: float = 0.1 
) -> Tuple[float, float]:
    """
    Trains the logistic model using Gradient Descent.
    Note: The learning rate usually needs to be higher in Logistic Regression (e.g., 0.1 or 1.0)
    because the sigmoid 'squashes' the values, making the gradients small.
    """
    alpha, beta = 0.0, 0.0
    
    for _ in range(epochs):
        g_alpha, g_beta = compute_log_gradients(x, y, alpha, beta)
        
        # Descent step
        alpha -= learning_rate * g_alpha
        beta -= learning_rate * g_beta
        
    return alpha, beta