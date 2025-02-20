import numpy as np
from scipy.special import expit, expm1
from scipy.linalg import inv

def gradient_descent(phi, group, lr=0.01):
	omega = np.random.rand(3, 1)
	eps=1e-2
	gradient = 100
	while np.sqrt(np.sum(gradient**2))>eps:
		gradient = phi.T@(group-1/(1+np.exp(-phi@omega)))
		omega = omega + lr*gradient

	return omega

def newton_method(phi, group, lr=0.01):
    omega = np.random.rand(3, 1)
    eps=1e-2
    N=len(phi)
    D=np.zeros((N, N))
    
    for i in range(N):
        D[i,i]=np.exp(-phi[i]@omega)/np.power(1+np.exp(-phi[i]@omega),2)
    
    H = phi.T@D@phi
    
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as error:
        print(str(error))
        print('Hessian matrix non invertible, switch to Gradient descent')
        return gradient_descent(phi, group)

    gradient = 100
    while np.sqrt(np.sum(gradient**2))>eps:
        gradient = H_inv@phi.T@(group-1/(1+np.exp(-phi@omega)))
        omega = omega + lr*gradient

    return omega




