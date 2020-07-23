import os
import pdb
import numpy as np
from numpy.linalg import eigh, inv
import scipy.stats as S
import autograd.numpy as anp
from sklearn.preprocessing import normalize
from autograd import grad, hessian
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt


def c(x, lamb):
    Lambda = np.diag(lamb)
    slx = np.repeat(np.sum(Lambda.dot(x), axis=0), repeats=x.shape[0], axis=0).reshape(x.shape)
    ldxslx = np.sqrt(Lambda).dot(x - slx)
    tr = np.trace(ldxslx.dot(ldxslx.T))

    return tr


def fd_c_grad_ij(X, marginal, sample, lamb, epsilon):
    e_ms = np.zeros(X.shape)
    e_ms[marginal, sample] = 1
    fd = (c(X + epsilon*e_ms, lamb) - c(X - epsilon*e_ms, lamb)) * e_ms / (2*epsilon)
    fd = fd[marginal]

    return fd


def fd_c_grad(X, marginal, lamb, epsilon):
    def e_alb(i, j):
        e = np.zeros(X.shape)
        e[i, j] = 1

        return e

    grad = np.array([(c(X + epsilon*e_mj, lamb) - c(X - epsilon*e_mj, lamb)) / (2*epsilon) for e_mj in [e_alb(marginal, j) for j in range(X.shape[1])]])

    return grad


def marginal_potential(X, mu, sigma):
    if not isinstance(X, float):
        return 0.5 * np.dot((X - mu), (X - mu).T) / (sigma**2)
    else:
        return 0.5 * ((X - mu) / sigma)**2


def schrodinger_potential(X, grad_V, laplacian_V):
    gVX = grad_V(X)
    if X.ndim > 1 and X.shape[1] > 1:
        potential = (np.dot(gVX, gVX.T) - 2*laplacian_V(X)) / 4
    else:
        potential = (gVX**2 - 2*laplacian_V(X)) / 4

    return potential


def discrete_laplacian(n, epsilon):
    laplace = np.diag(-2*np.ones((n,))) + np.diag(np.ones((n-1,)), k=1) + np.diag(np.ones((n-1,)), k=-1)
    laplace = laplace / (epsilon**2)

    return laplace


def schrodinger_laplacian(X, grad_V, laplacian_V, epsilon):
    pot_op = np.diag(schrodinger_potential(X, grad_V, laplacian_V))
    sl_op = -discrete_laplacian(X.shape[0], epsilon) + pot_op

    return sl_op


def inverse_schrodinger_kernel(X, V, grad_V, laplacian_V, epsilon, n_eigen=-1):
    if n_eigen == -1:
        n_eigen = X.shape[0] - 1

    L_S = schrodinger_laplacian(X, grad_V, laplacian_V, epsilon)

    eta, phi = eigh(L_S)
    eta, phi = eta[eta > 0], phi.T[eta > 0, :]
    eta, phi = eta[:n_eigen], phi[:n_eigen, :]

    rescale = np.repeat(np.exp(V(X)/2).reshape((1, phi.shape[1])), repeats=phi.shape[0], axis=0)
    phi = rescale * phi
    eta = inv(np.diag(eta))
    psi = np.dot(np.sqrt(eta), phi)

    K = np.dot(psi.T, psi).T

    return K


def fd_kernel_gradient(k, epsilon):
    grad_k = np.zeros(k.shape)

    grad_k[0] = (k[1, :] - k[0, :]) / epsilon
    for i in range(1, k.shape[0]-1):
        grad_k[i] = (k[i+1, :] - k[i-1, :]) / (2*epsilon)
    grad_k[-1] = (k[-1, :] - k[-2, :]) / epsilon

    return grad_k
