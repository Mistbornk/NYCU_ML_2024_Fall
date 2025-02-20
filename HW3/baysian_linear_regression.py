from utils import Polynomial_Basis_Data_Generator, Create_Design_Matrix
import numpy as np
from typing import List, Tuple
from copy import deepcopy

def Bayesian_Linear_Regression(precision: float, basis: int, variance: float, omega: List) -> Tuple[
    List[List[float]], float, np.ndarray, float, np.ndarray, float, np.ndarray]:
    """
    Bayesian linear regression
    :param basis: basis number of polynomial basis linear model
    :param variance: variance of polynomial basis linear model
    :param omega: weight vector of polynomial basis linear model
    :param precision: precision b for initial prior ~ N(0, b^-1 * I)
    :return: sample points, posterior mean, posterior covariance, tenth mean, tenth covariance, fiftieth mean and
    fiftieth covariance
    """

    count = 0
    points = []
    a = 1.0 / variance
    prior_mean = np.zeros((1, basis))
    tenth_mean = 0
    tenth_covariance = 0
    fiftieth_mean = 0
    fiftieth_covariance = 0
    while True:
        # Get a sample data point
        x, y = Polynomial_Basis_Data_Generator(basis, variance, omega)
        points.append([x, y])
        print(f'Add data point ({x}, {y}):\n')

        # Create design matrix from new data point
        design = Create_Design_Matrix(x, basis)

        # Get posterior mean and covariance
        # They are the mean and covariance of weight vector
        if not count:
            count += 1
            # First round
            # P(θ, D) ~ N(a(aA^T * A + bI)^-1 * A^T * y, (aA^T * A + bI)^-1)
            posterior_covariance = np.linalg.inv(a * design.T.dot(design) + precision * np.identity(basis))
            posterior_mean = a * posterior_covariance.dot(design.T) * y
        else:
            count += 1
            # N round
            # P(θ, D) ~ N((aA^T * A + S)^-1 * (aA^T * y + S * m), (aA^T * A + S)^-1)
            posterior_covariance = np.linalg.inv(a * design.T.dot(design) + np.linalg.inv(prior_covariance))
            posterior_mean = posterior_covariance.dot(
                a * design.T * y + np.linalg.inv(prior_covariance).dot(prior_mean))

        # Get marginalized mean and variance
        # They are the mean and variance of y
        marginalize_mean = design.dot(posterior_mean)
        marginalize_variance = variance + design.dot(posterior_covariance).dot(design.T)

        # Print posteriors
        print('Posterior mean:')
        for i in range(len(posterior_mean)):
            print(f'{posterior_mean[i, 0]:15.10f}')

        print('\nPosterior variance:')
        for row in range(len(posterior_covariance)):
            for col in range(len(posterior_covariance[row])):
                print(f'{posterior_covariance[row, col]:15.10f}', end='')
                if col < len(posterior_covariance[row]) - 1:
                    print(',', end='')
            print()

        # Print predictive distribution
        print(f'\nPredictive distribution ~ N({marginalize_mean[0, 0]:.5f}, {marginalize_variance[0, 0]:.5f})')
        print('--------------------------------------------------')

        # Get tenth and fiftieth posterior mean and covariance
        if count == 10:
            tenth_mean = deepcopy(posterior_mean)
            tenth_covariance = deepcopy(posterior_covariance)
        elif count == 50:
            fiftieth_mean = deepcopy(posterior_mean)
            fiftieth_covariance = deepcopy(posterior_covariance)

        # Break the loop if it converges
        if np.linalg.norm(posterior_mean - prior_mean) < 0.00001 and count > 50:
            break

        # Update prior
        prior_mean = posterior_mean
        prior_covariance = posterior_covariance

    return points, posterior_mean, posterior_covariance, tenth_mean, tenth_covariance, fiftieth_mean, fiftieth_covariance