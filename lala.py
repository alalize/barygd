import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def ratio(x):
    return x**2 / (2 + x - 2*np.sqrt(1 + x))


def ratio2(epsilon, dim=10):
    e = epsilon*np.ones(dim)
    fro = dim*epsilon**2
    tr = np.trace(2*np.eye(dim) + e - 2*sqrtm((np.eye(dim)+e)))
    ratio = fro / tr

    return ratio


x = np.linspace(0, 0.5, 256)

plt.grid()
plt.plot(x, [ratio2(v) for v in x], c='black')
plt.show()
