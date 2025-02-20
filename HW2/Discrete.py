import numpy as np

def Discreste_PixValProb(train_x, train_y):
	# return (label, digit, pix_value):(10,784,32) ndarray
	Prob = np.zeros((10, 28*28, 32))
	train_x_discrete = (train_x // 8).astype(int) 
	for label in range(10):
		# 對每一個 class 選出對應的 row 
		label_rows = train_x_discrete[train_y == label] # (label_row, digits, pixvalue_section)
		
		# 計算每個 digit 對應像素值區間的出現次數
		for dig in range(28*28):
			pixel_values = label_rows[:, dig] # 該 class 該 digit 的像素值區間
			# 該 class 的該 digit 像數值區間出現次數
			Prob[label, dig] = np.bincount(pixel_values, minlength=32)  # bincount 是計算每個像數值區間出現的次數，(32, 1)
	
	# 將計數轉換為機率
	Prob = Prob / Prob.sum(axis=2, keepdims=True)#機率，除以該 class 的該 digit 像數值區間出現次數總和
	return Prob

def Discrete_Classifier(num, PixValueProb, Prior, test_x, test_y):
	tol_err = 0
	log_prior = np.log(Prior)
    # 將 test_x 離散化到 0~31 的範圍
	test_x_discrete = (test_x // 8).astype(int) # (10000, 28*28) ele: 0~31
    
	# 計算所有樣本的 Posterior 機率
	for row in range(num): # 0-9999
        # 從 PixValueProb 中選取對應的 log(likelihood)，並在第 2 維（像素）上累加
		log_likelihoods = np.log(np.maximum(1e-4, PixValueProb[:, np.arange(28*28), test_x_discrete[row]]))
		Probs = log_likelihoods.sum(axis=1) + log_prior

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