from typing import List,Tuple 
from src.statistics import correlation, covariance, std_deviation, mean

def least_square_fit(x:List[float],y:List[float]) -> Tuple[float,float]:
    """
    Compute Linear Regression coefficients (alpha & Beta) 
    through least square fit

    Args:
        x(List[float]) features
    Returns:
        y(List[float]) target variable
    """

    # 1_st Step: Compute Least Square Beta
    #   $$\beta = \text{correlation}(x, y) \times \frac{\sigma_y}{\sigma_x}$$
    coeff_b = correlation(x,y) * (std_deviation(y)/std_deviation(x)) 

    # 2_nd Step: Compute Least Square Alpha
    #   $$\alpha = \bar{y} - \beta \bar{x}$$
    coeff_a = mean(y) - (coeff_b * mean(x))

    return (coeff_a,coeff_b)

def predict(alpha:float,beta:float,x_i:float)->float:
    """
    Predicts the value of target variable(y) 
    Based on the values of alpha,beta and i_th value of feature (x_i)
    y = alpha + beta * x

    Args:
        alpha(float): least squares alpha coefficient
        beta(float): least squares beta coefficient
        x_i(float): i_th value of the feature x
    
    Returns:
        float: target variable prediction
    """
    return alpha + beta * x_i

def compute_gradients(x: List[float], y: List[float], alpha: float, beta: float) -> Tuple[float, float]:
    """
    Compute the gradients of MSE Cost function in terms of alpha and beta.
    
    Returns:
        Tuple[float, float]: (grad_alpha, grad_beta)
    """
    n = len(x)
    
    # 1. Computing the predictions for all points: y_pred = alpha + beta * xi
    y_pred = [(alpha + beta * xi) for xi in x]
    
    # 2. Compute erros: error = y_pred - y_real
    error = [yp - yr for yp,yr in zip(y_pred,y)]
    
    # 3. Compute Alpha's gradient: Error sum * (2/n)
    alpha_gradient = sum(error) * (2/n) 

    # 4. Compute Beta's gradient: Erros's sum (erro * xi) * (2/n)
    beta_gradient = sum(
                        [error_i * x_i for error_i,x_i in zip(error,x)]
                        ) * (2/n)

    return (alpha_gradient,beta_gradient)

def gradient_descent_fit(
    x: List[float], 
    y: List[float], 
    epochs: int = 1000, 
    learning_rate: float = 0.01
) -> Tuple[float, float]:
    """
    Find alpha and beta minimizing quadratic error

    Args:
        x(List[float]): features
        y(List[float]): target variable 
        epochs: number of training iteractions 
        learning_rate: length of step
    Returns: 
        value of alpha and beta that minimizes cost function (MSE)
    """
    # 1. Inicialize alpha e beta com valores aleatórios (ou zeros)
    curr_alpha, curr_beta = 0.0, 0.0
    
    # 2. Loop pelas 'epochs' (quantas vezes vamos passar pelos dados)
    for epoch in range(epochs):
        # A. Calcule os gradientes usando a função que você criou acima
        grad_alpha, grad_beta = compute_gradients(x, y, curr_alpha, curr_beta)
        
        # B. Atualize os parâmetros (Passo do Gradiente)
        # novo_valor = valor_antigo - (learning_rate * gradiente)
        curr_alpha = curr_alpha - (learning_rate * grad_alpha)
        curr_beta = curr_beta - (learning_rate * grad_beta)
        
        # Opcional: A cada 100 epocas, printe o erro pra ver se tá diminuindo
        if epochs % 100 == 0:
            error = sum((curr_alpha + x_i*curr_beta - y_i) for x_i,y_i in zip(x,y))
            print(f"Error in the {epoch // 100}_ith iteraction: {error}")

    return curr_alpha, curr_beta
