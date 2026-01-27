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