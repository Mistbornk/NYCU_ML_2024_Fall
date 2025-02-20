import numpy as np
import math 

def Cal_Prior(train_y):
	prior = np.zeros(10)
	for label in range(10):
		# 每一個 label 的比率
		prior[label] = np.sum(train_y==label) / len(train_y)
	return prior

def Print_Image_byNumbers(PixValueProb, threshold):
	print('Imagination of numbers in Bayesian classifier:')
	for clss in range(10):
		print('{}:'.format(clss))
		for i in range(28):
			for j in range(28):
				print('+' if np.argmax(PixValueProb[clss, i*28+j])>=threshold else '0',end=' ')
			print()
		print()
	print()

def cal_mu(x):
	return np.mean(x, axis=0)

def cal_var(x, peudocont_var):
	var = np.var(x, axis=0)
	var[var==0] = peudocont_var
	return var


def gaussian_prob(x, mu, var):
	return (1/(np.sqrt(2*np.pi*var))) * np.exp(((-(x-mu)**2)/(2*var))) 

def facotrial(n):
    fac = 1
    for i in range(2, n + 1):
        fac *= i
    return fac

def C(N, m):
	if N-m < m:
		m = N-m
	res = 1
	for i in range(m):
		res *= N
		N -= 1
	res /= facotrial(m)
	return res

def B(a, b):
	return (Gamma(a)*Gamma(b))/Gamma(a+b)

def Gamma(a):
	return facotrial(a-1)



