import numpy as np
from Load_MNIST import load
from utils import *
from Discrete import *
from Continuous import *


if __name__=='__main__':
	train_x, train_y, test_x, test_y = load()
	
	toggle_bar=input('Toggle option [0 / 1] (discrete / continuous): ')

	if toggle_bar=='0':
		pixvalueProb = Discreste_PixValProb(train_x, train_y)
		prior = Cal_Prior(train_y)
		error_rate = Discrete_Classifier(len(test_y), pixvalueProb, prior, test_x, test_y)
		Print_Image_byNumbers(pixvalueProb, 16) 
		print('Error rate: {:.4f}'.format(error_rate))
	elif toggle_bar=='1':
		expected_var = 1000
		expected_prob = 1e-30 
		pixvalueProb = Continuous_PixValueProb(train_x, train_y, expected_var)
		prior = Cal_Prior(train_y)
		error_rate = Continuous_Classifier(len(test_y), pixvalueProb, prior, test_x, test_y, expected_prob)
		Print_Image_byNumbers(pixvalueProb, 128)
		print('Error rate: {:.4f}'.format(error_rate))
