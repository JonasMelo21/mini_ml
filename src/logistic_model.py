from src.activation import sigmoid
from typing import List, Tuple

def predict_probability(alpha: float, beta: float, x_i: float) -> float:
    """
    Returns the probability (0 to 1) of being the positive class.
    """
    # 1. Compute the z attribute for the sigmoid function
    z = alpha + beta * x_i 
    
    # 2. Pass to sigmoid
    return sigmoid(z)

def predict_class(alpha: float, beta: float, x_i: float, threshold: float = 0.5) -> int:
    """
    Returns 1 or 0 based on the probability.
    """
    return 1 if predict_probability(alpha, beta, x_i) > threshold else 0


def compute_log_gradients(
    x: List[float], 
    y: List[float], 
    alpha: float, 
    beta: float
) -> Tuple[float, float]:
    """
    Computes the gradient for Logistic Regression.
    The final formula is the same as Linear, but y_pred comes from the sigmoid.
    Gradient = (Prediction - Actual) * x
    """
    n = len(x)
    
    # 1. Prediction (Probability)
    # y_pred = sigmoid(alpha + beta * xi)
    predictions = [predict_probability(alpha, beta, xi) for xi in x]
    
    # 2. Error (Prediction - Actual)
    errors = [pred - target for pred, target in zip(predictions, y)]
    
    # 3. Gradients (Mean of errors * internal derivative)
    grad_alpha = sum(errors) * (1/n) # In logistic regression, sometimes we don't multiply by 2, but it depends on the convention. We'll use 1/n for simplicity.
    
    grad_beta = sum([err * xi for err, xi in zip(errors, x)]) * (1/n)
    
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