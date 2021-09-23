import pdb
import numpy as np
from numpy.linalg import inv, norm
from utils import norm2, gaussian_mixture_gradient
from fat_stick import fat_stick, dfat_stick
from sklearn.datasets import make_spd_matrix


def get_context(simul_params):
    grad_potentials = []
    means, stds, istds, eta = None, None, None, None

    if simul_params['laws'] == 'norms':
        means, stds = [-2, 6], [1, 1]
        grad_potentials.append(lambda x: (x - means[0]) / stds[0]**2)
        grad_potentials.append(lambda x: (x - means[1]) / stds[1]**2)
    elif simul_params['laws'] == 'norm-arctan':
        means, stds = [0, 0], [1, 1]
        grad_potentials.append(lambda x: (x - means[0]) / stds[0]**2)
        #grad_potentials.append(lambda x: np.sin(x) / np.cos(x)**3 - 2*np.tan(x))
        def grad_arctan(x):
            #if -np.pi/2 < x < np.pi:
            return np.sin(x)/np.cos(x)**3 - 2*np.tan(x)
            #print('(NA)  No gradient applied for arctan \# norm.')
            #pdb.set_trace()
            #return 0
        grad_potentials.append(grad_arctan)
    elif simul_params['laws'] == 'norm-exp':
        means, stds = [0, 0], [1, 1]
        grad_potentials.append(lambda x: (x - means[0]) / stds[0]**2)
        grad_potentials.append(lambda x: (1 + (np.log(x) - means[0])/(stds[0]**2)) / x)
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
        if not simul_params['orthogonal-normals']:
            eta_1, eta_2 = 0.5, 3
            means = [np.array([-6, 7]), np.array([5.5, 4])]
            stds = [np.eye(2), np.array([ [eta_1, 0], [0, eta_2] ])]
        else:
            epsilon = 1e-2
            means = [np.zeros(2), np.zeros(2)]
            stds = [np.diag([1, epsilon]), np.diag([epsilon, 1])]
        istds = [inv(s) for s in stds]
        grad_potentials.append(lambda x: istds[0].dot(x - means[0]))
        grad_potentials.append(lambda x: istds[1].dot(x - means[1]))
    elif simul_params['laws'] == '2d-n-norms':
        means = 6*np.random.multivariate_normal(np.zeros(simul_params['dimension']), np.eye(simul_params['dimension']), size=simul_params['n_marginals'])
        stds = [np.eye(simul_params['dimension']) + make_spd_matrix(simul_params['dimension']) for _ in range(simul_params['n_marginals'])]
        istds = [inv(s) for s in stds]
        for k, si in enumerate(istds):
            grad_potentials.append(lambda x, si=si: si.dot(x - means[k]))
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
    elif simul_params['laws'] == 'norm-mixture':
        norm_mean, norm_std = 3, 1
        mixture_means = np.array([-2.5, 1])
        mixture_stds = np.array([0.5, 1])
        mixture_weights = np.array([0.3, 0.7])
        means = [norm_mean, mixture_means, mixture_weights]
        stds = [norm_std, mixture_stds]
        grad_potentials.append(lambda x: (x - norm_mean) / norm_std**2)
        grad_potentials.append(gaussian_mixture_gradient(mixture_weights, mixture_means, mixture_stds))
    elif simul_params['laws'] == 'd-norms':
        import scipy.stats as stats

        means = [stats.binom.rvs(5, 0.5, size=simul_params['dimension']).squeeze() for _ in range(2)]
        std = np.diag(np.ones(simul_params['dimension']))
        std[-1, -1] = 10
        stds = [3*np.eye(simul_params['dimension']), std]
        istds = [inv(s) for s in stds]

        for k, istd in enumerate(istds):
            grad_potentials.append(lambda x,istd=istd,k=k: istd.dot(x - means[k])+1e-8)


    if simul_params['kernel-order'] == 2:
        kernel = lambda x, y: np.exp(-norm2(x-y)/2) + 1e-8
        dkernel = lambda x, y: (x-y)*np.exp(-norm2(x-y)/2) + 1e-8
    else:
        p = simul_params['kernel-order']
        kernel = lambda x, y: np.exp(-norm(x-y)**p)
        dkernel = lambda x, y: np.exp(-norm(x-y)**p) * (p*norm(x-y)**(p-2)) * (x-y)

    gd_context = {'k':kernel, 'dk':dkernel, 'dV':grad_potentials}
    return gd_context, grad_potentials, means, stds, istds, eta
