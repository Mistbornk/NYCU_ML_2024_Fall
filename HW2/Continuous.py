import numpy as np
from utils import *


def Continuous_PixValueProb(train_x, train_y, expected_var):
	Prob = np.zeros((10, 28*28, 256))

	# 將像素值從 0 到 255 向量化計算
	pixvalue = np.arange(256).reshape(-1, 1) #(256, 1)
	
	for label in range(10):
		label_row = train_x[train_y==label] #(label_rows, 784)

		# 每個 digit 的 mean 和 variance
		mu = cal_mu(label_row) #(784,)

		var = cal_var(label_row, expected_var) #(784,)

		# 計算每個像素點上的每個像素值的高斯分佈機率
		Prob[label] = gaussian_prob(pixvalue, mu, var).T #(784, 256)

	return Prob

def Continuous_Classifier(num, PixValueProb, Prior, test_x, test_y, expected_prob):
	tol_err = 0
	log_prior = np.log(Prior)
	for row in range(num):
		log_likelihood = np.log(np.maximum(expected_prob, PixValueProb[:, np.arange(28*28), test_x[row]])) # (10, 784)
		Probs = log_likelihood.sum(axis=1) + log_prior # (10,)

		Probs /= np.sum(Probs)

		print('Posterior (in log scale):')
		for label in range(10):
			print('{}: {}'.format(label, Probs[label]))
		pred = np.argmin(Probs) # 取最大機率的索引
		print('Prediction: {}, Ans: {}'.format(pred, test_y[row]))
		print()
		if pred != test_y[row]:
			tol_err += 1
	
	return tol_err / num
		

