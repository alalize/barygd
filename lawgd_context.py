import numpy as np
from numpy.linalg import eigh, inv, norm, det
from utils import lawgd_dk
from fat_stick import fat_stick, dfat_stick



def get_context(mesh, n_mesh, simul_params):
    potentials, grad_potentials, lap_potentials = [], [], []

    means, stds, istds, eta = None, None, None, None

    if simul_params['laws'] == 'norms':
        means = [-2, 4]
        stds = [1, 1]
        for k in range(2):
            potentials.append(lambda x,k=k: (x-means[k])**2 / (2*stds[k]**2))
            grad_potentials.append(lambda x: (x - means[k]) / stds[k]**2)
            lap_potentials.append(lambda x: 1/stds[k]**2)
    elif simul_params['laws'] == 'norm-arctan':
        means, stds = [0, 0], [1, 1]
        potentials.append(lambda x: x**2 / 2 + np.log(np.sqrt(2*np.pi)))
        potentials.append(lambda x: np.tan(x)**2 / 2 - np.log(1 + np.tan(x)**2) + np.log(np.sqrt(2*np.pi)))
        grad_potentials.append(lambda x: (x - means[0]) / stds[0]**2)
        grad_potentials.append(lambda x: np.sin(x) / np.cos(x)**3 - 2*np.tan(x))
        lap_potentials.append(lambda x: 1)
        lap_potentials.append(lambda x: (1+np.tan(x)**2)*(3*np.tan(x)**2-1))
    elif simul_params['laws'] == 'norm-exp':
        grad_potentials.append(lambda x: (x - mean_1) / std_1**2)
        grad_potentials.append(lambda x: (1 + (np.log(x) - mean_1)/(std_1**2)) / x)
    elif simul_params['laws'] == '3-norm':
        means, stds = [-3, 1, 5], [1, 0.6, 1]
        for k in range(3):
            grad_potentials.append(lambda x,k=k: (x-means[k]) / stds[k]**2) # THANK YOU, stackuder@recursive
    elif simul_params['laws'] == '3-norm-arctan-exp':
        mean, std = 0, 1
        grad_potentials.append(lambda x: (x - mean) / std**2)
        grad_potentials.append(lambda x: (np.tan(x) - mean)/(np.cos(x)**2 * std**2) - 2*np.tan(x))
        grad_potentials.append(lambda x: (np.log(x) - mean) / (x*std**2))
    elif simul_params['laws'] == '3-norm-arctan-norm':
        means, stds = [0, 4], [1, 2]
        grad_potentials.append(lambda x: (x - means[0]) / stds[0]**2)
        grad_potentials.append(lambda x: (np.tan(x) - means[0])/(np.cos(x)**2 * stds[0]**2) - 2*np.tan(x))
        grad_potentials.append(lambda x: (x - means[1]) / stds[1]**2)
    elif simul_params['laws'] == '2d-norm':
        eta_1, eta_2 = 0.5, 3
        means = [np.array([-6, 7]), np.array([5.5, 4])]
        stds = [np.eye(2), np.array([ [eta_1, 0], [0, eta_2] ])]
        istds = [inv(s) for s in stds]
        potentials.append(lambda x: 0.5*(x-means[0]).dot(istds[0]).dot(x-means[0]) + np.log(2*np.pi))
        potentials.append(lambda x: 0.5*(x-means[1]).dot(istds[1]).dot(x-means[1]) + np.log(2*np.pi))
        grad_potentials.append(lambda x: istds[0].dot(x - means[0]))
        grad_potentials.append(lambda x: istds[1].dot(x - means[1]))
        lap_potentials.append(lambda x: np.trace(istds[0])/2)
        lap_potentials.append(lambda x: np.trace(istds[1])/2)
    elif simul_params['laws'] == '2d-3-norm':
        eta_1, eta_2 = 0.5, 3
        means = [np.array([-6, 7]), np.array([5.5, 8]), np.array([4, -4])]
        stds = [np.eye(2), np.array([ [eta_1, 0], [0, eta_2] ]), \
            np.array([ [eta_1 + eta_2, eta_2 - eta_1], [eta_2 - eta_1, eta_1+eta_2] ])]
        istds = [inv(s) for s in stds]
        for k, istd in enumerate(istds):
            grad_potentials.append(lambda x,istd=istd,k=k: istd.dot(x - means[k]))
    elif simul_params['laws'] == '2d-sticks':
        eta = 20
        A = [ np.array([0, 0]), np.array([2, 1]) ]
        B = [ np.array([1, 1]), np.array([3, 0]) ]
        grad_potentials.append(dfat_stick(A[0], B[0], eta))
        grad_potentials.append(dfat_stick(A[1], B[1], eta))

    dk = [lawgd_dk(mesh, n_mesh, simul_params['dimension'], potentials[i], grad_potentials[i], \
        lap_potentials[i], simul_params['fd_step_size'], simul_params['n_eigen']) for i in range(len(potentials))]
    gd_context = {'dk':dk}

    return gd_context, potentials, grad_potentials, lap_potentials, means, stds, istds, eta
