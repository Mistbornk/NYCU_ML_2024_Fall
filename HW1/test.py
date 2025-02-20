from LSE import LSE
from NewtonMethod import NewtonMethod
from SteepestDescent import SteepestDescent
from utils import show_line, show_plot
import numpy as np




polynomial_basis_size = 3
LAMBDA = 10000

filepath='/Users/mistborn/Desktop/VScode/Python/ML/HW1/testfile.txt'
fp=open(filepath,'r')
line=fp.readline()
x=[]
y=[]
while line:
    temp1, temp2 = line.split(',')
    x.append(float(temp1))
    y.append(float(temp2))
    line=fp.readline()

x = np.asarray(x, dtype='float').reshape((-1, 1))
b = np.asarray(y, dtype='float').reshape((-1, 1))

A = np.empty((len(x),polynomial_basis_size))
for j in range(polynomial_basis_size):
    A[:,j] = np.power(x, j).reshape(-1)

x_LSE,loss_rlse = LSE(A,LAMBDA, b)
print('LSE: ')
show_line(x_LSE)
print('Total error: ', f'{loss_rlse}')
print()

print('Steepest Descent Method: ')
x_Steepest, loss_steepest = SteepestDescent(A, b, 0.0001, LAMBDA)
show_line(x_Steepest)
print('Total error: ', f'{loss_steepest}')
print()

x_Newton,loss_newton = NewtonMethod(A, b)
print('Newton\'s Method: ')
show_line(x_Newton)
print('Total error: ', f'{loss_newton}')

show_plot(x.reshape(-1), b.reshape(-1), x_LSE.reshape(-1), x_Steepest.reshape(-1), x_Newton.reshape(-1))