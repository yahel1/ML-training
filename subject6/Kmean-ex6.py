import numpy as np
import matplotlib.pyplot as plt

im_small = plt.imread('bird_small.tiff')
im_large = plt.imread('bird_large.tiff')
k = 16
N = len(im_small)
iy, ix = np.random.randint(0, N, k), np.random.randint(0, N, k)
means = im_small[iy, ix]
flat_im = im_small.reshape(N**2, 3)
c = np.zeros(N**2)
N_iter = 100
for i in range(N_iter):
    print(f'iter number: {i}')
    for pixel in range(N**2):
        c[pixel] = np.argmin(np.linalg.norm(flat_im[pixel] - means, axis=1))
    means_new = np.array([np.sum(flat_im[np.where(c == j)[0]], axis=0) / len(np.where(c == j)[0]) for j in range(k)])
    means_new[np.isnan(means_new)] = means[np.isnan(means_new)]
    means = np.copy(means_new)

flat_im = im_large.reshape(len(im_large)**2, 3)
im_large_clusters = np.zeros(np.shape(flat_im))
for pixel in range(len(im_large)**2):
    im_large_clusters[pixel, :] = means[np.argmin(np.linalg.norm(flat_im[pixel] - means, axis=1)), :]
im_large_clusters = im_large_clusters.reshape(len(im_large), len(im_large), 3)

plt.subplot(1, 2, 1)
plt.imshow(im_large)
plt.title('original picture')
plt.subplot(1, 2, 2)
plt.imshow(im_large_clusters/255)
plt.title('clustered picture')
