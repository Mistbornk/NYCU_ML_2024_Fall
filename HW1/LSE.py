import numpy as np
from utils import inverse, loss 

def LSE(A, LAMBDA, b):
    _, n = A.shape
    x = inverse(A.T@A + LAMBDA *np.eye(n))@A.T@b

    loss_value = loss(A, x, b)
    return x, loss_value