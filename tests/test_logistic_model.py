from src.logistic_model import predict_class, predict_probability, compute_log_gradients,fit

def test_logistic_prediction():
    # Imagine a model: if x > 5, it's class 1.
    # This happens if alpha = -5 and beta = 1.
    # z = -5 + 1*x.
    # If x=5 -> z=0 -> sigmoid(0)=0.5 (threshold)
    # If x=6 -> z=1 -> sigmoid(1)=0.73 -> Class 1
    alpha, beta = -5.0, 1.0
    # Probability test
    assert predict_probability(alpha, beta, x_i=6) > 0.5
    assert predict_probability(alpha, beta, x_i=4) < 0.5
    # Class test
    assert predict_class(alpha, beta, x_i=6) == 1
    assert predict_class(alpha, beta, x_i=4) == 0

def test_logistic_fit():
    # Clearly separable data
    # Group 0: Low values (1, 2, 3)
    # Group 1: High values (10, 11, 12)
    x = [1, 2, 3, 10, 11, 12]
    y = [0, 0, 0, 1, 1, 1]
    # Train
    alpha, beta = fit(x, y, epochs=2000, learning_rate=0.1)
    # Beta should be positive (the higher the X, the higher the chance of being 1)
    assert beta > 0
    # Alpha should be negative (to push the center of the sigmoid to the right)
    assert alpha < 0
    # Sanity test:
    # 2 should be class 0
    assert predict_class(alpha, beta, 2) == 0
    # 11 should be class 1
    assert predict_class(alpha, beta, 11) == 1
    # The turning point (decision boundary) should be between 3 and 10
    # Boundary: alpha + beta*x = 0  =>  x = -alpha/beta
    boundary = -alpha / beta
    assert 3 < boundary < 10