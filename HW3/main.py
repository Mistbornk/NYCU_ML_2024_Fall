from sequential_estimator import *
from baysian_linear_regression import *
from utils import draw_the_graph

if __name__=='__main__':
	state = int(input('Sequential Estimator / Baysian Linear regression [1 / 0] ? '))
	if state :
		mean = 3
		variance = 5
		Sequential_Estimator(mean, variance)
	else:
		b = 1
		n = 3
		a = 1
		w = [1, 2, 3]
		sample_points, posterior_mean, posterior_covariance, tenth_mean, tenth_covariance, fiftieth_mean, fiftieth_covariance = Bayesian_Linear_Regression(b, n, a, w)
		draw_the_graph(w, a, n, sample_points, posterior_mean, posterior_covariance, tenth_mean, tenth_covariance, fiftieth_mean, fiftieth_covariance)
