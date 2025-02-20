import numpy as np
from utils import inverse, loss, MSE

def NewtonMethod(A, b):
    m, n = A.shape
    x0 = np.random.rand(n, 1)
    eps = 100
    while eps > 1e-6:
        # hessian matrix = 2 A.T A
        x1 = x0 - inverse(2*A.T@A)@(2*A.T@A@x0-2*A.T@b)
        eps = MSE(x1, x0, n)
        x0 = x1
    
    loss_value = loss(A, x0, b)
    return x0, loss_value