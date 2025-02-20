import numpy as np
import matplotlib.pyplot as plt

def PLU_decomposition(A):
    # n = rows
    n = A.shape[0]
    # init P, L ,U
    U = A.copy()
    L = np.eye(n, dtype=np.double)
    P = np.eye(n, dtype=np.double)

    for i in range(n):
        # permute rows
        for j in range(i, n):
            if ~np.isclose(U[i, i], 0.0):
                break
            # swap j and j+1 row
            U[[j, j+1]] = U[[j+1, j]]
            P[[j, j+1]] = P[[j+1, j]]
        
        #Eliminate entries below i with row 
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]
    
    return P, L, U

# solve y let Ly = b
def forward_substitution(L, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=np.double);
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i] 
    return y

# solve x let Ux = y
def back_substitution(U, y):
    n = U.shape[0]
    x = np.zeros_like(y, dtype=np.double);
    x[-1] = y[-1] / U[-1, -1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i]
    return x 

# solve AA^ = I, LUA^ = PI
def inverse(A):
    n = A.shape[0]
    b = np.eye(n)
    A_inv = np.zeros((n, n))
    P, L, U = PLU_decomposition(A)
    for i in range(n):
        # solve Ly = Pb
        y = forward_substitution(L, np.dot(P, b[i, :]))
        # solve Ux = y
        A_inv[i, :] = back_substitution(U, y)
    return A_inv

def MSE(x1, x0, n):
    return abs(np.sum(np.square(x1-x0))/n)

def loss(A, x, b):
    return np.sum(np.square(A@x-b))

def show_line(x):
    x = x.reshape(-1)
    print('Fitting line: ',end='')
    # x^n-1 ~ x^1
    for i in range(len(x)-1, 0, -1):
        print(f'{x[i]}X^{i}', end='')
        if x[i-1]>0: print(' + ', end='')
        else: print(' ', end='')
    # x^0
    print(x[0])

def show_plot(x, b, x_LSE, x_Steepest, x_Newton):
    # LSE
    plt.subplot(3, 1, 1)
    plt.title('LSE')
    plt.plot(x, b,'ro')
    x_plot = np.linspace(min(x)-1, max(x)+1, 500)
    y_plot = np.zeros(x_plot.shape)
    for i in range(len(x_LSE)):
        y_plot += x_LSE[i]*np.power(x_plot, i)
    plt.plot(x_plot, y_plot, '-k')

    #Steepest Descent
    plt.subplot(3, 1, 2)
    plt.title('Steepest Descent method')
    plt.plot(x, b, 'ro')
    y_plot = np.zeros(x_plot.shape)
    for i in range(len(x_Steepest)):
        y_plot += x_Steepest[i]*np.power(x_plot, i)
    plt.plot(x_plot, y_plot, '-k')   

    #Newton
    plt.subplot(3, 1, 3)
    plt.title('newton\'s method')
    plt.plot(x, b, 'ro')
    y_plot = np.zeros(x_plot.shape)
    for i in range(len(x_Newton)):
        y_plot += x_Newton[i]*np.power(x_plot, i)
    plt.plot(x_plot, y_plot, '-k')

    # show
    plt.show()
