import os
import pdb
import numpy as np
from math import sqrt
from numpy.linalg import eigh, inv
import matplotlib.pyplot as plt


norm2 = lambda x: x.dot(x)
norm = lambda x: np.sqrt(norm2(x))


def c(x, lamb, convexifier=False):
    Lambda = np.diag(lamb)
    slx = np.repeat(np.sum(Lambda.dot(x), axis=0), repeats=x.shape[0], axis=0).reshape(x.shape)
    ldxslx = np.sqrt(Lambda).dot(x - slx)
    tr = np.trace(ldxslx.dot(ldxslx.T))

    if convexifier:
        tr += 0.5*np.trace(x.T.dot(x))

    return tr


def dc(x, lamb, convex=False):
    dx = np.zeros(x.shape)
    conv = 1 if convex else 0
    for k in range(x.shape[0]):
        dx[k] = 2*lamb[k]*(x[k] - np.sum([lamb[i]*x[i] for i in range(0, x.shape[0])])) \
            + x[k]*conv
    return dx


def lawgd_dk(mesh, dim, V, dV, ddV, epsilon, n_eigen=-1, domain=(-10, 10)):
    n = mesh.shape[0]
    if dim == 2:
        w = int(np.sqrt(n))
        ind = lambda x, epsilon=epsilon, w=w: w*int((x[0]-mesh[0,0])/epsilon) + int((x[1]-mesh[0,1])/epsilon)
    else:
        ind = lambda x, epsilon=epsilon, mesh=mesh: int((x-mesh[0])/epsilon)

    dV2 = np.sum(dV(mesh)**2, axis=-1) if dim == 2 else dV(mesh)**2
    lapV = np.array([ddV(mesh[i]) for i in range(n)])
    vs = np.diag(dV2 - 2*lapV)/4
    lap_eps = np.diag(-(4 if dim==2 else 2)*np.ones((n,)))+np.diag(np.ones((n-1,)), k=1)+np.diag(np.ones((n-1,)),k=-1)
    if dim == 2:
        lap_eps = lap_eps + np.diag(np.ones((n-w,)), k=w) + np.diag(np.ones((n-w,)), k=-w)
    lap = -lap_eps / epsilon**2 + vs

    eta_s, phi_s = eigh(lap)
    valid_indices = np.where(eta_s > 1e-2)[0]
    eta_s = eta_s[valid_indices]
    phi_s = phi_s.T[valid_indices]
    if n_eigen != -1:
        eta_s, phi_s = eta_s[:n_eigen], phi_s[:n_eigen]

    indic = -np.ones(mesh.shape)*(1e2)
    a, b = ind(domain[0]), ind(domain[1])
    indic[a:b] = 1
    rescale = np.repeat(np.exp(V(mesh)*indic/2).reshape((1,-1)), repeats=phi_s.shape[0], axis=0)
    rescale[:, :a] = 1
    rescale[:, b:] = 1
    phi = np.multiply(rescale, phi_s)
    eta = np.diag(1/eta_s)
    k = phi.T.dot(eta).dot(phi)

    if dim == 2:
        dk = np.zeros((k.shape[0], k.shape[1], mesh.shape[-1]))
        dk[0, :] = np.array((k[1, :] - k[0, :], k[w-1, :] - k[0, :])).T / epsilon
        for i in range(1, n-w-1):
            dk[i, :] = np.array((k[i+1,:] - k[i-1,:], k[i+w, :] - k[i, :])).T / (2*epsilon)
        dk[-1, :] = np.array((k[-1,:] - k[-2,:], k[-1, :] - k[-1-w+1, :])).T / epsilon
    else:
        dk = np.zeros(k.shape)
        dk[0] = (k[1,:] - k[0,:])/epsilon
        for i in range(1, n-1):
            dk[i] = (k[i+1,:]-k[i-1,:])/(2*epsilon)
        dk[-1] = (k[-1,:]-k[-2,:])/epsilon

    dkf = lambda x, y: dk[ind(x), ind(y)]

    return dkf


def lawgd_gradient(dk, L, N, i):
    grad = np.zeros(L.shape[-1])
    for k in range(N):
        grad = grad + dk(L[i], L[k])
    return -grad / N


def svgd_gradient(K, dK,  dV, L, N, i):
    grad = np.zeros(L.shape[-1])
    for k in range(N):
        grad = grad - dV(L[k]) * K(L[i], L[k]) + dK(L[i], L[k])
    return grad / N



#n = 50
#x = np.linspace(-1, 1, n)
##y = np.linspace(-1, 1, n)
#d = 1
##mesh = np.array([(x[i], y[j]) for j in range(n) for i in range(n)])
#mesh = x
#V = lambda x: np.sum(x**2, axis=-1) / 2
#dV = lambda x: x
#ddV = lambda x: 1
#lawgd_dk(mesh, d, V, dV, ddV, epsilon=(x.max() - x.min())/(n-1))
