import numpy as np
#from keras.datasets import mnist

def load():
	train_x_file=open('/Users/mistborn/Desktop/VScode/Python/ML/HW4/EMAlgorithm/train-images.idx3-ubyte_','rb')
	train_y_file=open('/Users/mistborn/Desktop/VScode/Python/ML/HW4/EMAlgorithm/train-labels.idx1-ubyte_','rb')

	# 處理 train_x 和 train_y
	train_x_file.read(16)  # 跳過 header
	train_y_file.read(8)   # 跳過 header

	# 使用 np.frombuffer 一次性讀取資料
	train_x = np.frombuffer(train_x_file.read(), dtype='uint8').reshape(60000, 28*28)
	train_y = np.frombuffer(train_y_file.read(), dtype='uint8')

	train_x = train_x.reshape(len(train_x), -1)
	train_x = np.asarray(train_x >= 128, dtype='uint8')

	return train_x, train_y
