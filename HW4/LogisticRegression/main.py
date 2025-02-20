from utils import *
from Logistic_Regression import *
from graph_utils import *

def default(d):
	if d == 0:
		return ( 50, 1, 1, 10, 10, 2, 2, 2, 2)
	elif d  == 1:
		return (50, 1, 1, 3, 3, 2, 2, 4, 4)

if __name__=='__main__':
	N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2 = default(1)
	D1, D2 = sampling(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2)
	phi = init_phi(D1, D2)
	group = init_group(N)

	gradient_descent_omega = gradient_descent(phi, group, 0.001)
	newton_method_omega = newton_method(phi, group, 0.001)

	print_results(N, phi, group, gradient_descent_omega, newton_method_omega, D1, D2)

	
