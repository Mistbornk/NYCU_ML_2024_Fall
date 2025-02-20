import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt

def Kernel(img):
	# 計算距離平方
	color_dist = np.sum((img[:, None, :] - img[None, :, :])**2, axis=-1)
	# 初始化空間座標
	coordinates = np.array([[i//100, i%100] for i in range(NUM_PIXEL)])
	# 計算距離平方
	spatial_distance = np.sum((coordinates[:, None, :] - coordinates[None, :, :])**2, axis=-1)
	# 計算 kernel
	img_kernel = np.exp(-GAMMA_S*spatial_distance) * np.exp(-GAMMA_C*color_dist)

	return img_kernel

def Kmeans(U):
	means, clusters = Init_Kmeans(U)

	converge = False
	iteration = 1
	
	while not converge:
		print(f'iteration: {iteration}')
		pre_clusters = clusters
		clusters = E_Step(U, means)
		means = M_Step(U, clusters)
		Savefig(clusters, iteration)
		converge = np.array_equal(clusters, pre_clusters)
		iteration += 1
	
	return clusters

def Init_Kmeans(U):
	if INIT_METHOD == 'r': # random
		centers = np.random.choice(NUM_PIXEL, NUM_CLUSTER, replace=False)
	elif INIT_METHOD == 'k': # k menas ++
		centers = np.zeros(NUM_CLUSTER, dtype=int)
		centers[0] = np.random.randint(NUM_PIXEL)
		for i in range(1, NUM_CLUSTER):
			distances = np.min([np.sum((U - U[c]) ** 2, axis=1) for c in centers[:i]], axis=0)
			probabilities = distances / np.sum(distances)
			centers[i] = np.random.choice(NUM_PIXEL, p=probabilities)
	else:
		raise ValueError('Wrong input for initial method !')
	
	means = U[centers]
	clusters = np.full(NUM_PIXEL, -1, dtype=int)
	clusters[centers] = np.arange(NUM_CLUSTER)

	return means, clusters

def E_Step(U, means):
    U = np.asarray(U, dtype=float)
    means = np.asarray(means, dtype=float)
    distances = np.linalg.norm(U[:, None, :] - means[None, :, :], axis=2)
    return np.argmin(distances, axis=1)

def M_Step(U, clusters):
    U = np.asarray(U, dtype=float)
    new_means = np.array([U[clusters == k].mean(axis=0) 
				if np.any(clusters == k) else np.zeros(U.shape[1]) for k in range(NUM_CLUSTER)], dtype=float)
    return new_means

def Laplacian(W):
    D = np.diag(W.sum(axis=1))
    L = D - W
    return L, D

def Normalize_Laplacian(L, D):
	sqrt_D = np.diag(1.0 / np.sqrt(np.diag(D)))
	L_norm = sqrt_D @ L @ sqrt_D
	return L_norm , sqrt_D 

def Eigen_Decomposition(L):
    eigenvalues, eigenvectors = np.linalg.eig(L)
    index = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, index]

    return eigenvectors

def Savefig(clusters, iteration):
    pixel = COLOR[clusters]
    pixel = np.reshape(pixel, (100, 100, 3))
    img = Image.fromarray(np.uint8(pixel))
    img.save(OUTPUT_DIR + '/%01d_%03d.png'%(NUM_CLUSTER, iteration), 'png')
    return

def Eigenspace(U, clusters):
    if NUM_CLUSTER == 2:
        plt.scatter(U[:, 0], U[:, 1], c=[EIGENSPACE_COLOR[c] for c in clusters])
    elif NUM_CLUSTER == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(U[:, 0], U[:, 1], U[:, 2], c=[EIGENSPACE_COLOR[c] for c in clusters])
    else:
        raise ValueError('Eigenspace can only be visualized for 2 or 3 clusters')
    plt.savefig(f'{OUTPUT_DIR}/eigenspace_{NUM_CLUSTER}.png')
    plt.show()
    

IMAGE_ID = int(input("Image id [1/2]: "))
NUM_CLUSTER = int(input("Number of cluster [2/3/4]: "))
MODE = input("Mode: normalized / ratio [n/r]? ")
INIT_METHOD = input("Initial method: random / kmeans++ [r/k]? ")
NUM_PIXEL = 10000
GAMMA_S = 0.001
GAMMA_C = 0.001
COLOR = np.array([[56, 207, 0], [64, 70, 230], [186, 7, 61], [245, 179, 66], [55, 240, 240]])
EIGENSPACE_COLOR = ['c', 'm', 'grey']
IMAGE_PATH = f'./data/image{IMAGE_ID}.png'
OUTPUT_DIR = f'./output/spectral_clustering/{MODE}/{INIT_METHOD}/image{IMAGE_ID}'

os.makedirs(OUTPUT_DIR, exist_ok=True)


img = np.asarray(Image.open(IMAGE_PATH).getdata())
W = Kernel(img)
L, D = Laplacian(W)

print(f'Image: {IMAGE_ID}, k: {NUM_CLUSTER}, mode: {MODE}, init_method: {INIT_METHOD}')
if MODE == 'r': # ratio cut
	Eigenvectors = Eigen_Decomposition(L)
	U = Eigenvectors[:, 1:1+NUM_CLUSTER].real
	clusters = Kmeans(U)
	if NUM_CLUSTER <= 3:
		Eigenspace(U, clusters)
elif MODE == 'n': # normalized
	L_norm, sqrt_D = Normalize_Laplacian(L, D)
	Eigenvectors = Eigen_Decomposition(L_norm)
	U = Eigenvectors[:, 1:1+NUM_CLUSTER].real
	T = sqrt_D @ U
	clusters = Kmeans(T)
	if NUM_CLUSTER <= 3:
		Eigenspace(T, clusters)
else:
	raise ValueError("Wrong input for Mode !")