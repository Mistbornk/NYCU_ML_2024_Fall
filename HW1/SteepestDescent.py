import numpy as np
from utils import loss
from utils import MSE

def SteepestDescent(A, b, lr=0.001, tol=1e-6, max_iter=100000, LAMBDA=0):
    m, n = A.shape
    x0 = np.random.rand(n, 1)
    iteration = 0
    while iteration < max_iter:
        gradient = (2*A.T@(A@x0-b))+ LAMBDA * np.sign(x0) 
        x1 = x0 - lr * gradient
        eps = np.linalg.norm(x1-x0)

        if eps < tol: break

        x0 = x1
        iteration += 1
    
    loss_value = loss(A, x0, b)
    return x0, loss_value

