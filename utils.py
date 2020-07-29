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
from numpy.polynomial.hermite_e import *


def c(x, lamb, convexifier=False):
    Lambda = np.diag(lamb)
    slx = np.repeat(np.sum(Lambda.dot(x), axis=0), repeats=x.shape[0], axis=0).reshape(x.shape)
    ldxslx = np.sqrt(Lambda).dot(x - slx)
    tr = np.trace(ldxslx.dot(ldxslx.T))

    if convexifier:
        tr += np.trace(x.T.dot(x))

    return tr


def fd_c_grad(X, marginal, lamb, epsilon, convexifier=False):
    def e_alb(i, j):
        e = np.zeros(X.shape)
        e[i, j] = 1

        return e

    grad = np.array([\
        (c(X + epsilon*e_mj, lamb, convexifier) - c(X - epsilon*e_mj, lamb, convexifier)) / (2*epsilon) \
            for e_mj in [e_alb(marginal, j) \
            for j in range(X.shape[1])]])

    return grad


def marginal_potential(X, mu, sigma):
    if not isinstance(X, float):
        pot = 0.5 * np.dot((X - mu), (X - mu).T) / (sigma**2)
    else:
        pot = 0.5 * ((X - mu) / sigma)**2

    return pot


def schrodinger_potential(X, grad_V, laplacian_V):
    gVX = grad_V(X)
    #if X.ndim > 1 and X.shape[1] > 1:
    #    potential = (np.dot(gVX, gVX.T) - 2*laplacian_V(X)) / 4
    #else:
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


def inverse_schrodinger_kernel(mesh, potential, grad_potential, laplacian_potential, epsilon, n_eigen=-1):
    if n_eigen == -1:
        n_eigen = mesh.shape[0] - 1

    L_S = schrodinger_laplacian(mesh, grad_potential, laplacian_potential, epsilon)

    eta_S, phi_S = eigh(L_S)
    eta_S[eta_S < 0] = np.min(eta_S[eta_S > 0])
    phi_S = phi_S.T
    eta_S, phi_S = eta_S[:n_eigen], phi_S[:n_eigen]

    rescale = np.repeat(np.exp(potential(mesh)/2).reshape((1, -1)), repeats=phi_S.shape[0], axis=0)
    phi = np.multiply(rescale, phi_S)
    eta = inv(np.diag(np.sqrt(eta_S)))
    psi = np.dot(eta, phi)

    K = np.dot(psi.T, psi)

    return K


def fd_kernel_gradient(k, epsilon):
    grad_k = np.zeros(k.shape)

    grad_k[0] = (k[:, 1] - k[:, 0]) / epsilon
    for i in range(1, k.shape[1]-1):
        grad_k[i] = (k[:, i+1] - k[:, i-1]) / (2*epsilon)
    grad_k[-1] = (k[:, -1] - k[:, -2]) / epsilon

    return grad_k
