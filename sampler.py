import os
import argparse
import pdb
import numpy as np
from progress.bar import Bar
from numpy.linalg import eigh, inv
import scipy.stats as S
import autograd.numpy as autograd_np
from sklearn.preprocessing import normalize
from autograd import grad, hessian
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
import subprocess
import utils
import datetime


np.random.seed(0)
now_str = str(datetime.datetime.now()).replace(' ', '_')

just_lawgd = False


# mesh parameters
n_mesh = 128
mesh_x_min, mesh_x_max = -6, 6
#mesh = np.ogrid[mesh_x_min:mesh_x_max:complex(0, n_mesh)]
mesh = np.linspace(mesh_x_min, mesh_x_max, num=n_mesh)


# simulation parameters
bary_coords = [1, 0]
simul_params = {
    'n_schemes': 1,
    'n_eigenvalues': -1,
    'langevin_perturbation': False,
    'langevin_factor': 8e-4,
    'n_iterations': 20000,
    'initial_domain': (4.5, 5),
    'fd_step_size': (mesh_x_max - mesh_x_min) / (n_mesh - 1),
    'tfd_step_size': 1e-6,
    'gd_step_size': 1e-2,
    'regularization': 1,
    'n_samples': 50,
    'bary_coords': bary_coords,
    'n_marginals': len(bary_coords)
}
exp_name = now_str + ''.join(['{}:{}_'.format(k[:4], simul_params[k]) for k in simul_params.keys()])

print('NAME:', exp_name)
print('n_mesh: {}\t fd_step_size: {}'.format(n_mesh, simul_params['fd_step_size']))


# definition of marginals
mean_1, mean_2 = 0, 3
std_1, std_2 = 1, 1
marginal_1 = S.norm.pdf(mesh, loc=mean_1, scale=std_1)
marginal_2 = S.norm.pdf(mesh, loc=mean_2, scale=std_2)
bary_12 = S.norm.pdf(mesh, loc=bary_coords[0]*mean_1 + mean_2*bary_coords[1], scale=1)
avg_12 = bary_coords[0]*marginal_1 + bary_coords[1]*marginal_2
mixture_plt = (marginal_1 + marginal_2)/2


# computing kernels for the marginals and their gradients
potential_1 = lambda x: utils.marginal_potential(x, mean_1, std_1)
potential_2 = lambda x: utils.marginal_potential(x, mean_2, std_2)

#mixture_12 = lambda x: -autograd_np.log(autograd_np.exp(-(x-mean_1)**2/2) + autograd_np.exp(-(x-mean_2)**2/2))
#potential_1 = mixture_12

gradient_potential_1 = egrad(potential_1)
gradient_potential_2 = egrad(potential_2)


laplacian_potential_1 = np.vectorize(lambda x: float(hessian(potential_1)(x)))
laplacian_potential_2 = np.vectorize(lambda x: float(hessian(potential_2)(x)))

kernel_1 = utils.inverse_schrodinger_kernel(mesh, np.vectorize(potential_1), \
    gradient_potential_1, laplacian_potential_1, simul_params['fd_step_size'], \
        n_eigen=simul_params['n_eigenvalues'])
kernel_2 = utils.inverse_schrodinger_kernel(mesh, np.vectorize(potential_2), \
    gradient_potential_2, laplacian_potential_2, simul_params['fd_step_size'], \
        n_eigen=simul_params['n_eigenvalues'])

gradient_kernel_1 = utils.fd_kernel_gradient(kernel_1, simul_params['fd_step_size'])
gradient_kernel_2 = utils.fd_kernel_gradient(kernel_2, simul_params['fd_step_size'])
kernel_gradients = [gradient_kernel_1, gradient_kernel_2]


def lawgd(mesh, kernel_gradient, simul_params):
    def lawgd_iteration(L, it):
        L_new = np.zeros(L.shape)

        mesh_projections = [np.argmin(np.abs(L[j] - mesh)) for j in range(simul_params['n_samples'])]
        grads = []

        for i in range(simul_params['n_samples']):
            grad = (simul_params['gd_step_size'] / simul_params['n_samples']) \
                    * np.sum(kernel_gradient[mesh_projections[i], mesh_projections], axis=0)

            if np.abs(grad) > 2*simul_params['gd_step_size']: grad = np.sign(grad)*simul_params['gd_step_size']

            L_new[i] = L[i] - grad

            grads.append(grad)
        
        return L_new, grads

    initial_domain_start, initial_domain_end = simul_params['initial_domain'][0], simul_params['initial_domain'][1]
    initial_domain_length = initial_domain_end - initial_domain_start
    X = S.uniform.rvs(loc=initial_domain_start, scale=initial_domain_length, size=simul_params['n_samples']).reshape((-1, 1))
    X_init = np.copy(X)

    plt.ion()
    for iteration in range(simul_params['n_iterations']):
        X, grads = lawgd_iteration(X, iteration)

        if iteration % 500 == 0:
            plt.clf()
            plt.title('iteration {}'.format(iteration))
            plt.grid()
            plt.scatter(X_init, np.zeros(simul_params['n_samples']), c='b', alpha=0.05)
            plt.scatter(X, np.zeros(simul_params['n_samples']), c='r', marker='x')
            plt.scatter(X, grads, c='g', alpha=0.1, marker='o')
            plt.xlim(mesh.min(), mesh.max())
            plt.draw()
            plt.pause(0.001)
    plt.ioff()
    plt.close()
    
    return X, X_init


if just_lawgd:
    swarm, swarm_init = lawgd(mesh, kernel_gradients[0], simul_params)
    valid_samples = swarm[np.where(np.logical_and(swarm < mesh.max(), swarm > mesh.min()))[0]]
    lawgd_density = S.gaussian_kde(valid_samples.reshape((-1,)))
    print('valid_samples lawgd:', valid_samples.size)

    from scipy.optimize import curve_fit

    def linear_model(x, a, b):
        return a*x + b

    #points = S.norm.rvs(loc=0, scale=1, size=1000)
    points = valid_samples
    hist, bins = np.histogram(points, bins=50, density=True)
    centers = np.array([0.5*(bins[i] + bins[i+1]) for i in range(50)])


    y = np.log(hist)
    valids = list(set(list(range(y.size))).difference(set(np.where(np.isinf(y))[0].tolist())))
    x = 0.5*(centers-mean_1)**2
    x, y = x[valids], y[valids]

    ab_opt, pcov = curve_fit(linear_model, x, y)
    p_err = round(np.abs((-1) - ab_opt[0]), 2)
    print('fit parameters: (slope, ord) = {}'.format(ab_opt))
    print('distnace to normal:', p_err)

    t = np.linspace(x.min(), x.max())
    y_fit = np.vectorize(lambda x: linear_model(x, ab_opt[0], ab_opt[1]))(t)


    fig, axs = plt.subplots(nrows=1, ncols=2)

    axs[0].set_title('fit to normal, slope: {}, fiterr: {}'.format(round(ab_opt[0], 2), p_err))
    axs[0].scatter(x, y)
    axs[0].plot(t, y_fit)
    axs[0].grid()

    axs[1].set_title('lawgd samples')
    axs[1].scatter(swarm_init, np.zeros(swarm_init.size), c='b', label='initial samples')
    axs[1].scatter(valid_samples, np.zeros(valid_samples.size), c='r', label='lawgd samples')
    axs[1].plot(mesh, lawgd_density.pdf(mesh), c='b', alpha=0.7, label='lawgd density')
    axs[1].plot(mesh, marginal_1, c='b', alpha=0.2, label='target density')
    axs[1].hist(valid_samples, density=True, color='r', alpha=0.1, bins='auto')
    #axs[1].plot(mesh, mixture_plt, c='b', alpha=0.2, label='target density')
    axs[1].grid()
    axs[1].legend()

    plt.savefig('./img/{}.png'.format(exp_name))
    plt.show()

    quit(0)


# simulation procedure in one dimension
def bary_gd(mesh, kernel_gradients, simul_params):
    def lawgd_iteration(L):
        L_new = np.copy(L)

        for alpha in range(simul_params['n_marginals']):
            mesh_projections = [np.argmin(np.abs(L[alpha, j] - mesh)) for j in range(simul_params['n_samples'])]

            for i in range(simul_params['n_samples']):
                L_new[alpha, i] = L[alpha, i] - \
                    simul_params['gd_step_size'] \
                        * simul_params['regularization'] \
                        * simul_params['bary_coords'][alpha] \
                        * np.sum(kernel_gradients[alpha][mesh_projections[i], mesh_projections], axis=0) \
                        / simul_params['n_samples']

                if simul_params['langevin_perturbation']:
                    L_new[alpha, i] = L_new[alpha, i] + simul_params['langevin_factor']*np.sqrt(2*simul_params['gd_step_size'])*np.random.normal()
        
        return L_new


    def bary_iteration(X, L):
        for alpha in range(simul_params['n_marginals']):
            L[alpha, 0] = X[alpha]
        L_new = lawgd_iteration(L)
        X_new = np.copy(X)

        for alpha in range(simul_params['n_marginals']):
            mesh_projection_alpha = np.argmin(np.abs(X[alpha] - mesh))
            c_gradient_alpha = utils.fd_c_grad(X, alpha, simul_params['bary_coords'], simul_params['tfd_step_size'])

            X_new[alpha] = -simul_params['gd_step_size']*c_gradient_alpha + L_new[alpha, 0]
        
        return X_new, L_new

    initial_domain_start, initial_domain_end = simul_params['initial_domain'][0], simul_params['initial_domain'][1]
    initial_domain_length = initial_domain_end - initial_domain_start
    X = S.uniform.rvs(loc=initial_domain_start, scale=initial_domain_length, size=simul_params['n_marginals']).reshape((-1, 1))
    L = S.uniform.rvs(loc=initial_domain_start, scale=initial_domain_length, size=simul_params['n_marginals']*simul_params['n_samples']) \
        .reshape((simul_params['n_marginals'], simul_params['n_samples'], 1))

    X_init = np.copy(X)

    plt.ion()

    for iteration in range(simul_params['n_iterations']):
        X, L = bary_iteration(X, L)

        if iteration % 500 == 0:
            plt.clf()
            plt.title('iteration {}'.format(iteration))
            plt.grid()
            plt.scatter(L[0, :], np.zeros(L.shape[1]), c='r', marker='x')
            plt.scatter(L[1, :], np.zeros(L.shape[1]), c='r', marker='x')
            plt.xlim(mesh.min(), mesh.max())
            plt.draw()
            plt.pause(0.001)

    plt.ioff()
    
    for alpha in range(simul_params['n_marginals']):
        L[alpha, 0] = X[alpha]

    return X_init, X


# execute simulation and estimate the density of the barycenter using a gaussian kernel estimator
init_samples, swarm = bary_gd(mesh, kernel_gradients, simul_params)
for it in range(simul_params['n_schemes']-1):
    init_new, swarm_new = bary_gd(mesh, kernel_gradients, simul_params)

    np.append(init_samples, init_new, axis=0)
    np.append(swarm, swarm_new, axis=0)

valid_samples = swarm[np.where(np.logical_and(swarm < 10*mesh.max(), swarm > 10*mesh.min()))[0]]
bary_density = S.gaussian_kde(valid_samples.reshape((-1,)))

print('n_valid_samples:', valid_samples.size)


# plot marginals and expected density, the coupling samples and the estimated density for the barycenter
plt.clf()
plt.title(r'barygd -- $\alpha = {0}$, niter = {1}'.format(simul_params['regularization'], simul_params['n_iterations']))

plt.plot(mesh, marginal_1, label='marginal 1', linestyle='--', color='gray', alpha=0.3)
plt.plot(mesh, marginal_2, label='marginal 2', linestyle='--', color='gray', alpha=0.3)
plt.plot(mesh, bary_12, label='barycenter', linestyle='--', color='blue', alpha=0.5)
plt.plot(mesh, avg_12, label='average', linestyle='--', color='blue', alpha=0.4)

plt.plot(mesh, bary_density.pdf(mesh), label='estimated', color='black')
plt.scatter(init_samples[:, 0], np.zeros(simul_params['n_marginals']), marker='o', label='init samples', c='blue', alpha=0.7)
plt.scatter(swarm[:, 0], np.zeros(swarm.shape[0]), marker='x', label='bary samples', c='black')
plt.hist(swarm[:, 0], bins=50, density=True, alpha=0.2, color='red')

plt.xlim(mesh.min(), mesh.max())

plt.grid()
plt.legend()
plt.show()
