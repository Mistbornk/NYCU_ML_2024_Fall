import numpy as np
from matplotlib import pyplot as plt
from typing import List, Tuple, Union

# Based on central limit theorem and Irwin-Hall
def Univariate_Gaission_Data_Generater(mean: float, variance: float):
	return (np.sum(np.random.uniform(0, 1, 12)) - 6) * np.sqrt(variance) + mean

def Polynomial_Basis_Data_Generator(basis, variance, omega:List):
	x = np.random.uniform(-1, 1)
	y = Univariate_Gaission_Data_Generater(0, variance)

	for power, w in enumerate(omega):
		y += w * np.power(x, power)

	return x, y

def Create_Design_Matrix(x: float, basis: int) -> np.ndarray:
	design = np.zeros((1, basis))

	for i in range(basis):
		design[0, i] =  np.power(x, i)
	
	return design

def draw_the_graph(weight_vector: List[float], variance: float, basis: int, points: List[List[float]],
                   posterior_mean: float, posterior_covariance: np.ndarray, tenth_mean: float,
                   tenth_covariance: np.ndarray, fiftieth_mean: float, fiftieth_covariance: np.ndarray) -> None:
    """
    Draw the graph
    :param weight_vector: weight vector of the ground truth
    :param variance: variance of polynomial basis linear model
    :param basis: basis number of polynomial basis linear model
    :param points: sample points
    :param posterior_mean: converged posterior mean
    :param posterior_covariance: converged posterior covariance
    :param tenth_mean: tenth posterior mean
    :param tenth_covariance: tenth posterior covariance
    :param fiftieth_mean: fiftieth posterior mean
    :param fiftieth_covariance: fiftieth posterior covariance
    :return: None
    """
    x = np.linspace(-2.0, 2.0, 100)
    points = np.transpose(points)

    # Ground truth
    plt.subplot(221)
    plt.title('Ground truth')
    f = np.poly1d(np.flip(weight_vector))
    y = f(x)
    draw_lines(x, y, variance)

    # Predict result
    plt.subplot(222)
    plt.title('Predict result')
    f = np.poly1d(np.flip(np.reshape(posterior_mean, basis)))
    y = f(x)
    var = np.zeros(100)
    for i in range(100):
        design = Create_Design_Matrix(x[i], basis)
        var[i] = variance + design.dot(posterior_covariance).dot(design.T)[0, 0]
    plt.scatter(points[0], points[1], s=1)
    draw_lines(x, y, var)

    # After 10 times
    plt.subplot(223)
    plt.title('After 10 times')
    f = np.poly1d(np.flip(np.reshape(tenth_mean, basis)))
    y = f(x)
    var = np.zeros(100)
    for i in range(100):
        design = Create_Design_Matrix(x[i], basis)
        var[i] = variance + design.dot(tenth_covariance).dot(design.T)[0, 0]
    plt.scatter(points[0][:10], points[1][:10], s=1)
    draw_lines(x, y, var)

    # After 50 times
    plt.subplot(224)
    plt.title('After 50 times')
    f = np.poly1d(np.flip(np.reshape(fiftieth_mean, basis)))
    y = f(x)
    var = np.zeros(100)
    for i in range(100):
        design = Create_Design_Matrix(x[i], basis)
        var[i] = variance + design.dot(fiftieth_covariance).dot(design.T)[0, 0]
    plt.scatter(points[0][:50], points[1][:50], s=1)
    draw_lines(x, y, var)

    plt.tight_layout()
    plt.show()


def draw_lines(x: np.ndarray, y: np.ndarray, variance: Union[float, np.ndarray]) -> None:
    """
    Draw predict line and two lines with variance
    :param x: x coordinates of the points
    :param y: y coordinates of the points
    :param variance: y variance
    :return: None
    """
    plt.plot(x, y, color='k')
    plt.plot(x, y + variance, color='r')
    plt.plot(x, y - variance, color='r')
    plt.xlim(-2.0, 2.0)
    plt.ylim(-15.0, 25.0)





