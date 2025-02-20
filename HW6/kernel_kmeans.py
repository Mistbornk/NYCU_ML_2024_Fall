import math
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import os


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

def Kernel_Kmeans(img_kernel, clusters, C):
    new_clusters = np.zeros(NUM_PIXEL, dtype=int)
    pq = Sum_pq(img_kernel, clusters, C)
    for pixel in trange(NUM_PIXEL):
        distances = np.zeros(NUM_CLUSTER)
        for cluster in range(NUM_CLUSTER):
            distances[cluster] = img_kernel[pixel, pixel] - Sum_n(img_kernel[pixel], clusters, C, cluster) + pq[cluster]
        new_clusters[pixel] = np.argmin(distances)
    new_C = Cal_C(new_clusters)

    return new_clusters, new_C

def Sum_n(pixel_kernel, clusters, C, k):
	return 2 / C[k] * np.sum(pixel_kernel[clusters == k])

def Sum_pq(kernel, clusters, C):
	sum = np.zeros(NUM_CLUSTER)
	for k in range(NUM_CLUSTER):
		mask = clusters == k
		sum[k] = np.sum(kernel[mask][:, mask]) / (C[k]**2)
	return sum

def Cal_C(clusters):
	return np.bincount(clusters, minlength=NUM_CLUSTER)

def Init_Cluster(kernel, centers):
	# 初始化每個 pixel 的 cluster
	clusters = np.full(NUM_PIXEL, -1, dtype=int)
	for pixel in range(NUM_PIXEL):
		if pixel in centers:
			clusters[pixel] = np.where(centers == pixel)[0][0]
		else:
			clusters[pixel] = np.argmin(kernel[pixel][centers])

	C = Cal_C(clusters)
	Savefig(clusters, 0)

	return clusters, C

def Init(kernel):
	if MODE == 'r': # random mode
		# 隨機選取 k 個不重複的 pixel
		centers = np.random.choice(NUM_PIXEL, NUM_CLUSTER, replace=False)
	elif MODE == 'k': # k means++
		centers = np.zeros(NUM_CLUSTER, dtype=int)
		centers[0] = np.random.randint(NUM_PIXEL)
		for i in range(1, NUM_CLUSTER):
			distances = np.min(kernel[centers[:i]], axis=0)
			probabilities = distances / np.sum(distances)
			centers[i] = np.random.choice(NUM_PIXEL, p=probabilities)

	else:
		raise ValueError("Wrong input for Mode !")

	clusters, C = Init_Cluster(kernel, centers)
	return clusters, C

def Savefig(clusters, iteration):
    pixel = COLOR[clusters]
    pixel = np.reshape(pixel, (100, 100, 3))
    img = Image.fromarray(np.uint8(pixel))
    img.save(OUTPUT_DIR + '/%01d_%03d.png'%(NUM_CLUSTER, iteration), 'png')

def main():
	print(f'Image: {IMAGE_ID}, k: {NUM_CLUSTER}, mode: {MODE}')
	img = np.asarray(Image.open(IMAGE_PATH).getdata())

	kernel = Kernel(img)
	clusters, C = Init(kernel)
	converge = False
	iteration = 1

	while not converge:
		print(f'iteration: {iteration}')
		pre_clusters = clusters
		clusters, C = Kernel_Kmeans(kernel, clusters, C)
		Savefig(clusters, iteration)
		converge = np.array_equal(clusters, pre_clusters)
		iteration += 1

IMAGE_ID = int(input("Image id [1/2]: "))
NUM_CLUSTER = int(input("Number of cluster [2/3/4]: "))
MODE = input("Mode: random / kmeans++ [r/k]? ")
NUM_PIXEL = 10000
GAMMA_S = 0.001
GAMMA_C = 0.001
COLOR = np.array([[56, 207, 0], [64, 70, 230], [186, 7, 61], [245, 179, 66], [55, 240, 240]])
IMAGE_PATH = f'./data/image{IMAGE_ID}.png'
OUTPUT_DIR = f'./output/kernel_kmeans/{MODE}/image{IMAGE_ID}'

os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
	main()
