import numpy as np
from libsvm.svmutil import *
from concurrent.futures import ThreadPoolExecutor, as_completed

def Load_Data():
	x_train_filepath = "./data/X_train.csv"
	y_train_filepath = "./data/Y_train.csv"
	x_test_filepath = "./data/X_test.csv"
	y_test_filepath = "./data/Y_test.csv"
	def Load_File(filepath, is_label=False):
		data = []
		with open(filepath) as f:
			for line in f:
				if is_label:
					data.append(int(line.strip()))
				else:
					data.append(np.array(line.split(',')).astype(float))
		return np.array(data)

	X_train = Load_File(x_train_filepath, is_label=False)
	Y_train = Load_File(y_train_filepath, is_label=True)
	X_test = Load_File(x_test_filepath, is_label=False)
	Y_test = Load_File(y_test_filepath, is_label=True)

	return X_train, Y_train, X_test, Y_test

def Grid_Search(X_train, Y_train, kernel_type):
    print(f'\n{["linear", "polynomial", "RBF", "sigmoid"][kernel_type]} kernel:')
    
    cost = np.array([0.01, 0.1, 1, 10])
    degree = np.array([2, 3])
    gamma = np.array([1/784, 0.01, 0.1, 1])
    coefficient = np.array([-5, 0, 5, 10])
    
    optimal_parameters = f'-s 0 -v 3 -q'
    optimal_accuracy = 0.0

    # Helper function to evaluate a single parameter combination
    def evaluate(parameters):
        """計算當前參數組合的準確率"""
        accuracy = svm_train(Y_train, X_train, parameters)
        return (parameters, accuracy)

    futures = []
    # 限制最多使用 3 個 thread
    max_threads = 3
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        if kernel_type == 0:  # linear
            for c in cost:
                parameters = f'-s 0 -t {kernel_type} -c {c} -q -v 3'
                futures.append(executor.submit(evaluate, parameters))

        elif kernel_type == 1:  # polynomial
            for c in cost:
                for d in degree:
                    for g in gamma:
                        for r in coefficient:
                            parameters = f'-s 0 -t {kernel_type} -c {c} -d {d} -g {g} -r {r} -q -v 3'
                            futures.append(executor.submit(evaluate, parameters))

        elif kernel_type == 2:  # RBF
            for c in cost:
                for g in gamma:
                    parameters = f'-s 0 -t {kernel_type} -c {c} -g {g} -q -v 3'
                    futures.append(executor.submit(evaluate, parameters))

        elif kernel_type == 3:  # sigmoid
            for c in cost:
                for g in gamma:
                    for r in coefficient:
                        parameters = f'-s 0 -t {kernel_type} -c {c} -g {g} -r {r} -q -v 3'
                        futures.append(executor.submit(evaluate, parameters))

        # Process results as they complete
        for future in as_completed(futures):
            parameters, accuracy = future.result()
            print(parameters)
            if accuracy > optimal_accuracy:
                optimal_parameters = parameters
                optimal_accuracy = accuracy
    
    optimal_parameters = optimal_parameters[:-5]    #remove -v
    print(f"optimal accuracy: {optimal_accuracy}")
    print(f"optimal parameters: {optimal_parameters}")
    
    return optimal_parameters


def Linear_Kernel(u, v):
	return u @ v.T

def RBF_Kernel(u, v, gamma=1/784):
	dist = np.sum(u**2, axis=1, keepdims=True) + np.sum(v**2, axis=1) - 2 * u @ v.T
	return np.exp(-gamma * dist)

Task = int(input('Please choose a Task to run (1, 2, 3): '))
X_train, Y_train, X_test, Y_test = Load_Data()

if Task == 1:
	# 測試 linear
	model = svm_train(Y_train, X_train, f'-t {0} -q')
	result = svm_predict(Y_test, X_test, model)
	# 測試 polynomial
	model = svm_train(Y_train, X_train, f'-t {1} -q')
	result = svm_predict(Y_test, X_test, model)
	# 測試 RBF kernel
	model = svm_train(Y_train, X_train, f'-t {2} -q')
	result = svm_predict(Y_test, X_test, model)

elif Task == 2:
	# 進行 grid serch 並將結果輸出到檔案
	with open('SVM_Task2.txt', 'w') as f:
		# linear, polynomial, RBF kernel, sigmoid
		for kernel_type in range(4):
			optimal_parameters = Grid_Search(X_train, Y_train, kernel_type)
			model = svm_train(Y_train, X_train, optimal_parameters)
			result = svm_predict(Y_test, X_test, model)
			f.write(f'Kernel {kernel_type}: {optimal_parameters}\n')

elif Task == 3:
	# 使用自定義核函數 (linear + RBF)
	train_kernel = Linear_Kernel(X_train, X_train) + RBF_Kernel(X_train, X_train)
	train_kernel = np.hstack((np.arange(1, train_kernel.shape[0] + 1).reshape(-1, 1), train_kernel))

	test_kernel = Linear_Kernel(X_test, X_train) + RBF_Kernel(X_test, X_train)
	test_kernel = np.hstack((np.arange(1, test_kernel.shape[0] + 1).reshape(-1, 1), test_kernel))
    
	model = svm_train(Y_train, train_kernel, '-t 4 -q')
	result = svm_predict(Y_test, test_kernel, model)

else:
	print('Invalid input!')