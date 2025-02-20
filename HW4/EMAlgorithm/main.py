from dataload import load
from utils import *

if __name__=='__main__':
	train_x, train_y = load()
	Lambda = init_lambda()
	PixVal = init_PixVal()

	last_diff, diff, count, eps = 1000, 100, 0, 1

	GroundTruth_distribution =  get_pixvalueProb_discrete(train_x, train_y)

	while abs(last_diff-diff) > eps and count<=15:

		# E-step
		W = Update_Posterior(train_x, Lambda, PixVal)
		
		# M-step
		Lambda_new = Update_Lambda(W)
		PixVal_new = Update_Distribution(train_x, W)

		# cal diff
		last_diff = diff
		diff = np.sum(np.abs(Lambda-Lambda_new)) + np.sum(np.abs(PixVal-PixVal_new))
		Lambda = Lambda_new
		PixVal = PixVal_new
		
		# plot
		class_order = perfect_matching(GroundTruth_distribution, PixVal)
		plot(PixVal, class_order, threshold=0.35)

		print()
		print(f"No. of Iteration: {count+1}, Difference: {diff}")
		print('------------------------------------------------------------')
		count += 1
	print('------------------------------------------------------------')

	maximum = np.argmax(W, axis=1)
	unique, counts = np.unique(maximum, return_counts=True)

	class_order = perfect_matching(GroundTruth_distribution, PixVal)

	plot(PixVal, class_order, threshold=0.35)
	confusion_matrix(train_y, maximum, class_order)
	print_error_rate(count, train_y, maximum, class_order)