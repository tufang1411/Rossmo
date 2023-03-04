import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Rumus Rossmo
def rossmo_formula(x, y, c, w):
    """
    x: ndarray of shape (n, 3) containing the x and y coordinates of the n crime locations
       as well as a constant value (e.g. 0 or 1) in a third column
    y: ndarray of shape (m, 2) containing the x and y coordinates of the m grid points
    c: float representing the distance decay parameter
    w: ndarray of shape (n,) containing the weights of the n crime locations
    """
    dists = cdist(x[:, :2], y, metric='euclidean')
    weights = w[:, np.newaxis] / (1 + c*dists)
    return np.sum(weights, axis=0)

# Titik Lokasi Pembunuhan
crime_locations = np.array([[39.6, 187.5, 0], [38, 186.5, 0], [40, 188, 0], [75.5, 195, 0], [83.5, 179, 0], [113, 192.5, 0], [115, 191.5, 0], [130.5, 280.7, 0], [139, 160, 0], [139.9, 160.4, 0], [140.5, 159.6, 0], [149.7, 298.9, 0], [162.2, 247, 0], [169.3, 337, 0], [172.6, 265, 0], [175.6, 204, 0], [191.5, 171.5, 0], [194, 253.9, 0], [204, 229.4, 0], [205.4, 224, 0], [222.5, 231, 0], [241.6, 176, 0], [247.6, 188, 0], [262, 316.5, 0], [264, 229, 0], [276, 213, 0], [322.5, 178, 0], [412, 132, 0], [456.5, 179, 0]])

# Area 500x500
grid_points = np.mgrid[0:500:500j, 0:500:500j].reshape(2,-1).T

# Hitung Menggunakan Rossmo Untuk Nilai Setiap Kisi / Grid
rossmo_values = rossmo_formula(crime_locations, grid_points, c=0.07, w=np.ones(crime_locations.shape[0]))

# Mengubah Nilai Rossmo Ke Dalam Grid
rossmo_grid = rossmo_values.reshape(500, 500)

# Heatmap Dari Nilai Rossmo
plt.imshow(rossmo_grid, cmap='inferno', interpolation='bicubic', origin='lower')

# Menitikkan Titik Lokasi Pembunuhan
plt.scatter(crime_locations[:,0], crime_locations[:,1], c='red', s=15)

# Titik Rumah Pembunuh
plt.scatter(176, 243, c='#000000', s=20)

plt.colorbar()

plt.show()
