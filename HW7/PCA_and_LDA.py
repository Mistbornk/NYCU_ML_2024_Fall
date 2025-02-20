import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import trange

import os
import numpy as np

def Load_Pgm(path):
    pgms, labels = [], []
    
    def Read_Pgm(path):
        with open(path, 'rb') as f:
            assert f.readline() == b'P5\n'
            f.readline()  # 跳過註解行
            width, height = map(int, f.readline().split())
            assert int(f.readline()) <= 255
            pgm = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width))
        return pgm.flatten()
    
    for name in os.listdir(path):
        data = Read_Pgm(os.path.join(path, name))
        pgms.append(data)
        labels.append(int(name[7:9]))
    
    return np.array(pgms), np.array(labels)

def Resize_Pgm(data):
    num_pgms = data.shape[0]
    resized_height, resized_width = 77, 65
    block_size = 3
    pgm_compress = np.zeros((num_pgms, resized_height*resized_width))
    
    for pgm in range(num_pgms):
        reshaped_pgm = data[pgm].reshape(231, 195)
        compressed_pgm = reshaped_pgm.reshape(resized_height, block_size, resized_width, block_size).mean(axis=(1, 3))
        pgm_compress[pgm] = compressed_pgm.flatten()
        
    return pgm_compress

def Load_and_Resize_Pmg(train_path, test_path):
    train_pgm, train_label = Load_Pgm(train_path)
    train_pgm = Resize_Pgm(train_pgm)
    test_pgm, test_label = Load_Pgm(test_path)
    test_pgm = Resize_Pgm(test_pgm)
    
    return train_pgm, train_label, test_pgm, test_label   
    
def PCA(x):
	covariance = np.cov(x.T)
	eigenvalue, eigenvector = np.linalg.eigh(covariance)
	eigenvector = eigenvector[:, np.argsort(-eigenvalue)]
	W = eigenvector[:, :25] / np.linalg.norm(eigenvector[:, :25], axis=0)
	return W

def LDA(data, label):
	n = data.shape[1]
	Sw, Sb = np.zeros((n, n)), np.zeros((n, n))
	mean = np.mean(data, axis=0)

	for subject in trange(1, 16):
		x = data[label == subject]
		mean_cluster = np.mean(x, axis=0)
		Sw += (x - mean_cluster).T @ (x - mean_cluster)
		Sb += x.shape[0] * (mean_cluster - mean).T @ (mean_cluster - mean)

	SwSb = np.linalg.pinv(Sw) @ Sb
	eigenvalue, eigenvector = np.linalg.eigh(SwSb)
	eigenvector = eigenvector[:, np.argsort(-eigenvalue)]
	W = eigenvector[:, :25] / np.linalg.norm(eigenvector[:, :25], axis=0)
	
	return W

def Kernel_PCA(x, kernel_type):
	'''
	kernel type: 'Linear', 'RBF', 'Linear+RBF', 'Polynomial', 'Sigmoid'
	'''
	kernels = {
		'RBF': lambda x, y: RBF_Kernel(x, y),
		'Linear': lambda x, y: Linear_Kernel(x, y),
		'Linear+RBF': lambda x, y: Linear_Kernel(x, y)+RBF_Kernel(x, y),
		'Polynomial': lambda x, y: Polynomial_Kernel(x, y),
        'Sigmoid': lambda x, y: Sigmoid_Kernel(x, y)
	}
	if kernel_type not in kernels:
		raise ValueError("Invalid kernel type !")
    
	kernel = kernels[kernel_type](x, x)
	eigenvalue, eigenvector = np.linalg.eigh(kernel)
	eigenvector = eigenvector[:, np.argsort(-eigenvalue)]
	W = eigenvector[:, :25] / np.linalg.norm(eigenvector[:, :25], axis=0)   
    
	return W, kernel

def Kernel_LDA(x, kernel_type):
	'''
	kernel type: 'Linear', 'RBF', 'Linear+RBF', 'Polynomial', 'Sigmoid'
	'''
	kernels = {
		'RBF': lambda x, y: RBF_Kernel(x, y),
		'Linear': lambda x, y: Linear_Kernel(x, y),
		'Linear+RBF': lambda x, y: Linear_Kernel(x, y)+RBF_Kernel(x, y),
		'Polynomial': lambda x, y: Polynomial_Kernel(x, y),
        'Sigmoid': lambda x, y: Sigmoid_Kernel(x, y)
	}
	if kernel_type not in kernels:
		raise ValueError("Invalid kernel type !")
	subject_N = 9
	kernel = kernels[kernel_type](x, x)
	L = np.ones((x.shape[0], x.shape[0])) / subject_N 
	Sw = kernel @ kernel
	Sb = kernel @ L @ kernel
	SwSb = np.linalg.pinv(Sw) @ Sb
	eigenvalue, eigenvector = np.linalg.eigh(SwSb)
	eigenvector = eigenvector[:, np.argsort(-eigenvalue)]
	W = eigenvector[:, :25] / np.linalg.norm(eigenvector[:, :25], axis=0)
	
	return W, kernel

def Linear_Kernel(u, v):
	return u @ v.T

def RBF_Kernel(u, v, gamma=1e-9):
	dist = np.sum(u**2, axis=1, keepdims=True) + np.sum(v**2, axis=1) - 2 * u @ v.T
	return np.exp(-gamma * dist)

def Polynomial_Kernel(u, v, degree=2, coef0=1):
    return (u @ v.T + coef0) ** degree

def Sigmoid_Kernel(u, v, alpha=1e-9, coef0=1):
    return np.tanh(alpha * (u @ v.T) + coef0)

def Show_EigenFisher_Face(W, filname):
    fig, axes = plt.subplots(5, 5, figsize=(5, 5))
    for i, ax in enumerate(axes.flat):
        pgm = W[:, i].reshape(77, 65)
        ax.axis('off')
        ax.imshow(pgm, cmap='gray')
    plt.tight_layout()
    fig.savefig(f'{filname}.png')
    #plt.show()

def Reconstruct_Face(W, data, filename):
    choice = np.random.choice(data.shape[0], 10, replace=False)
    fig, axes = plt.subplots(2, 10, figsize=(10, 2))

    for i, idx in enumerate(choice):
        original = data[idx].reshape(77, 65)
        reconstructed = (original.flatten() @ W @ W.T).reshape(77, 65)

        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].axis('off')

        axes[1, i].imshow(reconstructed, cmap='gray')
        axes[1, i].axis('off')

    plt.tight_layout()
    fig.savefig(f'{filename}.png')
    #plt.show()

def Predict_Pgm(train_pgm, train_label, test_pgm, test_label, W):
    k, accuracy = 5, 0
    xW_train, xW_test = train_pgm @ W, test_pgm @ W

    size = xW_test.shape[0]
    for test_vec, true_label in zip(xW_test[:size], test_label[:size]):
        distances = np.linalg.norm(xW_train - test_vec, axis=1)
        neighbors = np.argsort(distances)[:k]
        prediction = np.bincount(train_label[neighbors]).argmax()
        accuracy += (true_label == prediction)

    print(f'Accuracy: {accuracy / size * 100:.2f}%')

def Predict_Kernel_Pgm(train_pgm, train_label, test_pgm, test_label, W, train_kernel, kernel_type):
	'''
	kernel type: 'Linear', 'RBF', 'Linear+RBF', 'Polynomial', 'Sigmoid'
	'''
	kernels = {
		'RBF': lambda x, y: RBF_Kernel(x, y),
		'Linear': lambda x, y: Linear_Kernel(x, y),
		'Linear+RBF': lambda x, y: Linear_Kernel(x, y)+RBF_Kernel(x, y),
		'Polynomial': lambda x, y: Polynomial_Kernel(x, y),
        'Sigmoid': lambda x, y: Sigmoid_Kernel(x, y)
	}
	if kernel_type not in kernels:
		raise ValueError("Invalid kernel type !")

	k, accuracy = 5, 0
	test_kernel = kernels[kernel_type](test_pgm, train_pgm)
	xW_train, xW_test = train_kernel @ W, test_kernel @ W

	size = xW_test.shape[0]
	for test_vec, true_label in zip(xW_test[:size], test_label[:size]):
		distances = np.linalg.norm(xW_train - test_vec, axis=1)
		neighbors = np.argsort(distances)[:k]
		prediction = np.bincount(train_label[neighbors]).argmax()
		accuracy += (true_label == prediction)

	print(f'Accuracy: {accuracy / size * 100:.2f}%')

def Shift_to_Center(train_pgm, test_pgm):
	mean_pgm = np.mean(train_pgm, axis=0)
	shift_train = train_pgm - mean_pgm
	shift_test = test_pgm - mean_pgm

	return shift_train, shift_test

if __name__ == "__main__":
	MODE = input("Select a mode (PCA / LDA / kernel PCA / kernel LDA): \n")

	train_pgm, train_label, test_pgm , test_label = Load_and_Resize_Pmg('Training/', 'Testing/')

	if MODE == "PCA":
		W_PCA = PCA(train_pgm)
		Show_EigenFisher_Face(W_PCA, 'PCA_Eigenface')
		Reconstruct_Face(W_PCA, train_pgm, 'PCA_Reconstruct')
		Predict_Pgm(train_pgm, train_label, test_pgm, test_label, W_PCA)
	elif MODE == "LDA":
		W_LDA = LDA(train_pgm, train_label)
		Show_EigenFisher_Face(W_LDA, 'LDA_Eigenface')
		Reconstruct_Face(W_LDA, train_pgm, 'LDA_Reconstruct')
		Predict_Pgm(train_pgm, train_label, test_pgm, test_label, W_LDA)
	elif MODE == "kernel PCA":
		kernel_type = 'Polynomial'
		shift_train, shift_test = Shift_to_Center(train_pgm, test_pgm)
		W, kernel = Kernel_PCA(shift_train, kernel_type)
		Predict_Kernel_Pgm(shift_train, train_label, shift_test, test_label, W, kernel, kernel_type)
	elif MODE == "kernel LDA":
		kernel_type = 'Sigmoid'
		shift_train, shift_test = Shift_to_Center(train_pgm, test_pgm)
		W, kernel = Kernel_LDA(shift_train, kernel_type)
		Predict_Kernel_Pgm(shift_train, train_label, shift_test, test_label, W, kernel, kernel_type)
	else:
		raise ValueError(f"Wrong input !\n")

     

    