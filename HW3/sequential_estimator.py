from utils import Univariate_Gaission_Data_Generater

# Sequential estimate the mean and variance using Welford's online algorithm
def Sequential_Estimator(mean: float, variance: float):
	print(f'Data point source function: N({mean}, {variance})\n')

	count = 0
	last_mean, last_variance = mean, variance
	while True:
		new_x = Univariate_Gaission_Data_Generater(mean, variance)
		count += 1
		
		# initial
		if count==1:
			current_mean = new_x
			population_variance = 0
			sample_variance = 0
			M2 = 0
		else:
			delta1 = new_x - current_mean
			current_mean += delta1 / count
			delta2 = new_x - current_mean
			M2 += delta1 * delta2
			population_variance = M2 / (count)
			sample_variance = M2 / (count-1)
		

		print(f'Add data point: {new_x}')
		print(f'Mean = {current_mean}\tVariance = {sample_variance}')
		if abs(current_mean - last_mean) < 0.0001 and abs(sample_variance - last_variance) < 0.0001:
			break
		last_mean, last_variance = current_mean, sample_variance 
