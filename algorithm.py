import scipy

def regression(X, y, W):
    return (X.T * W * X).I*(X.T * W * y)
