import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def Load_Data():
	file_path = "./data/input.data"
	X, Y = [], []
	with open(file_path) as f:
		for line in f:
			x, y = map(float, line.split())
			X.append(x)
			Y.append(y)	
	return np.array(X), np.array(Y)

def RationalQuadratic(x1, x2, alpha, length_scale, kernel_variance):
	kernel = kernel_variance * ((1 + ((x1-x2)**2) / (2*alpha*length_scale**2))**(-alpha))
	return kernel

def Create_Covariance_Matrix(X, beta, alpha, length_scale, kernel_variance):
    num_data = len(X)
    Covariace = np.zeros((num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            Covariace[i, j] = RationalQuadratic(X[i], X[j], alpha, length_scale, kernel_variance)
        # 加上噪聲項到對角線元素
        Covariace[i, i] += 1 / beta
    return Covariace  

def Predict(X, Y, covariance, beta, alpha, length_scale, kernel_variance, sample_size):
    x_sample = np.linspace(-60, 60, sample_size)
    inv_cov = inv(covariance)
    mean = np.zeros(sample_size)
    variance = np.zeros(sample_size)
    
    for sample_idx, x_star in enumerate(x_sample):
        # 計算與所有訓練點的核函數值向量
        kernel = np.array([RationalQuadratic(x_i, x_star, alpha, length_scale, kernel_variance) for x_i in X])
        kernel_star = RationalQuadratic(x_star, x_star, alpha, length_scale, kernel_variance) + 1/beta
        
		 # GP 均值和方差公式
        mean[sample_idx] = kernel.T @ inv_cov @ Y
        variance[sample_idx] = kernel_star - (kernel.T @ inv_cov @ kernel)
        
    return mean, variance

def Gaussian_Process(X, Y, Beta, Alpha, LengthScale, kernel_variance, Sample_Size):
	Covariance = Create_Covariance_Matrix(X, Beta, Alpha, LengthScale, kernel_variance)
	mean, variance = Predict(X, Y, Covariance, Beta, Alpha, LengthScale, kernel_variance, Sample_Size)
    
	return mean , variance

def Negative_Marginal_Log_Likelihood(theta, X, Y, beta):
    alpha, length_scale, kernel_variance = theta
    n = X.shape[0]
    
	# 計算協方差矩陣
    covariance = Create_Covariance_Matrix(X, beta, alpha, length_scale, kernel_variance)
    
	# 邊際對數似然公式
    log_likelihood = 0.5 * np.log(np.linalg.det(covariance))
    log_likelihood += 0.5 * Y.T @ np.linalg.inv(covariance) @ Y
    log_likelihood += 0.5 * n * np.log(2 * np.pi)
    
    return log_likelihood

def Plot_Result(ax, X, Y, mean, variance, sample_size, title):
    x_sample = np.linspace(-60, 60, sample_size)
    interval = 1.96 * (variance ** 0.5)

    ax.scatter(X, Y, color='b', label='Data')
    ax.plot(x_sample, mean, color='k', label='Mean')
    ax.plot(x_sample, mean + interval, color='r', linestyle='--', label='95% CI')
    ax.plot(x_sample, mean - interval, color='r', linestyle='--')
    ax.fill_between(x_sample, mean + interval, mean - interval, color='pink', alpha=0.3)
    
    ax.set_title(title)
    ax.legend()
    ax.grid()

# define parameters
Beta = 5
Alpha = 1
LengthScale = 1
Kernel_Variance = 1.725272
Sample_Size = 1000

# Load data
X, Y = Load_Data()

# GP (Task1)
mean, variance = Gaussian_Process(X, Y, Beta, Alpha, LengthScale, Kernel_Variance, Sample_Size)

# GP optimized (task2)
initial_params = [Alpha, LengthScale, Kernel_Variance]
opt = minimize(Negative_Marginal_Log_Likelihood, initial_params, args=(X, Y, Beta))
Alpha_opt, LengthScale_opt, Kernel_Variance_opt = opt.x
mean_opt, variance_opt = Gaussian_Process(X, Y, Beta, Alpha_opt, LengthScale_opt, Kernel_Variance_opt, Sample_Size)
print(f"{'Parameter':<15}{'Initial':>15}{'Optimized':>15}")
print("-" * 45)
print(f"{'alpha:':<15}{Alpha:>15.6f}{Alpha_opt:>15.6f}")
print(f"{'lengthscale:':<15}{LengthScale:>15.6f}{LengthScale_opt:>15.6f}")
print(f"{'kernel_variance:':<15}{Kernel_Variance:>15.6f}{Kernel_Variance_opt:>15.6f}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

Plot_Result(axes[0], X, Y, mean, variance, Sample_Size, "Initial GP")
Plot_Result(axes[1], X, Y, mean_opt, variance_opt, Sample_Size, "Optimized GP")

plt.tight_layout()
plt.show()