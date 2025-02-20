import numpy as np

def load():
	train_x_file=open('/Users/mistborn/Desktop/VScode/Python/ML/HW2/train-images.idx3-ubyte_','rb')
	train_y_file=open('/Users/mistborn/Desktop/VScode/Python/ML/HW2/train-labels.idx1-ubyte_','rb')
	test_x_file=open('/Users/mistborn/Desktop/VScode/Python/ML/HW2/t10k-images.idx3-ubyte_','rb')
	test_y_file=open('/Users/mistborn/Desktop/VScode/Python/ML/HW2/t10k-labels.idx1-ubyte_','rb')

	# 處理 train_x 和 train_y
	train_x_file.read(16)  # 跳過 header
	train_y_file.read(8)   # 跳過 header

	# 使用 np.frombuffer 一次性讀取資料
	train_x = np.frombuffer(train_x_file.read(), dtype='uint8').reshape(60000, 28*28)
	train_y = np.frombuffer(train_y_file.read(), dtype='uint8')

	# 處理 test_x 和 test_y
	test_x_file.read(16)  # 跳過 header
	test_y_file.read(8)   # 跳過 header

	# 一次性讀取 test_x 和 test_y
	test_x = np.frombuffer(test_x_file.read(), dtype='uint8').reshape(10000, 28*28)
	test_y = np.frombuffer(test_y_file.read(), dtype='uint8')

	return train_x, train_y, test_x, test_y