from src.activation import sigmoid

def predict_probability(alpha: float, beta: float, x_i: float) -> float:
    """
    Retorna a probabilidade (0 a 1) de ser a classe positiva.
    """
    # 1. Computing sigmoid's function z attribute
    z = alpha + beta * x_i 
    
    # 2. Pass to sigmoid
    return sigmoid(z)

def predict_class(alpha: float, beta: float, x_i: float, threshold: float = 0.5) -> int:
    """
    Retorna 1 ou 0 baseado na probabilidade.
    """
    return 1 if predict_probability(alpha,beta,x_i) > threshold else 0