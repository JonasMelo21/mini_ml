from src.logistic_model import predict_class, predict_probability

def test_logistic_prediction():
    # Imagine um modelo que diz: se x > 5, é classe 1.
    # Isso acontece se alpha = -5 e beta = 1.
    # z = -5 + 1*x. 
    # Se x=5 -> z=0 -> sigmoide(0)=0.5 (limiar)
    # Se x=6 -> z=1 -> sigmoide(1)=0.73 -> Classe 1
    
    alpha, beta = -5.0, 1.0
    
    # Teste de probabilidade
    assert predict_probability(alpha, beta, x_i=6) > 0.5
    assert predict_probability(alpha, beta, x_i=4) < 0.5
    
    # Teste de classe
    assert predict_class(alpha, beta, x_i=6) == 1
    assert predict_class(alpha, beta, x_i=4) == 0