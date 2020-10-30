import os
import argparse
import pdb
import numpy as np
from progress.bar import Bar
from numpy.linalg import eigh, inv, norm
import scipy.stats as S
import autograd.numpy as anp
from sklearn.preprocessing import normalize
from autograd import grad, hessian
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
import subprocess
import utils
import datetime
from scipy.optimize import curve_fit


np.random.seed(0)
now_str = str(datetime.datetime.now()).replace(' ', '_')
draw_freq = 5


#x = np.array([[1], [2]])
#print(utils.dc(x, [0.3, 0.7]))


# mesh parameters
n_mesh = 1024
mesh_x_min, mesh_x_max = -14, 14
mesh = np.linspace(mesh_x_min, mesh_x_max, num=n_mesh)


# simulation parameters
l = 0.5
bary_coords = [l, 1-l]

simul_params = {
    'n_schemes': 1,
    'n_eigenvalues': -1,
    'langevin_perturbation': False,
    'langevin_factor': 1,
    'g_conv': True,
    'n_iterations': 320,
    'initial_domain': (0, 3),
    'fd_step_size': (mesh_x_max - mesh_x_min) / (n_mesh - 1),
    'tfd_step_size': 1e6,
    'gd_step_size': 1e-1,
    'regularization': 1000,
    'n_samples': 100,
    'bary_coords': bary_coords,
    'n_marginals': len(bary_coords),
    'laws': 'norm',
    'kernel_width': 1
}


exp_name = 'bar_' + now_str + ''.join(['{}:{}_'.format(k[:4], simul_params[k]) for k in simul_params.keys()])
print('NAME:', exp_name)
print('n_mesh: {}\t fd_step_size: {}'.format(n_mesh, simul_params['fd_step_size']))


# definition of marginals
if simul_params['laws'] == 'norm':
    mean_1, mean_2 = -2, 6 
    std_1, std_2 = 1, 1
    marginal_1 = S.norm.pdf(mesh, loc=mean_1, scale=std_1)
    marginal_2 = S.norm.pdf(mesh, loc=mean_2, scale=std_2)
    bary = S.norm.pdf(mesh, loc=bary_coords[0]*mean_1 + mean_2*bary_coords[1], scale=1)
    avg_12 = bary_coords[0]*marginal_1 + bary_coords[1]*marginal_2
    mixture_plt = (marginal_1 + marginal_2)/2
    mean_bary = bary_coords[0] * mean_1 + bary_coords[1] * mean_2
elif simul_params['laws'] in ['norm-arctan', 'norm-exp']:
    mean_1, std_1 = 0, 1
elif simul_params['laws'] == 'exp':
    loc_1, loc_2 = -7, 7
    scale_1, scale_2 = 1, 1

    marginal_1 = S.laplace.pdf(mesh, loc=loc_1, scale=scale_1)
    marginal_2 = S.laplace.pdf(mesh, loc=loc_2, scale=scale_2)

    def FinvFi(mesh, loc, scale):
        return S.laplace.ppf(S.laplace.cdf(mesh, loc=loc, scale=scale), loc=loc_1, scale=scale_1)
    t_1 = FinvFi(mesh, loc_1, scale_1)
    t_2 = FinvFi(mesh, loc_2, scale_2)
    f_1 = S.laplace.pdf(t_1, loc=loc_1, scale=scale_1)
    f_2 = S.laplace.pdf(t_2, loc=loc_2, scale=scale_2)
    f_12 = S.laplace.pdf(t_2, loc=loc_1, scale=scale_1)
    bary = bary_coords[0]*marginal_1 + bary_coords[1]*(f_2/f_12)*marginal_2


# computing kernels for the marginals and their gradients
if simul_params['laws'] == 'norm':
    dpot_1 = lambda x: (x - mean_1) / std_1**2
    dpot_2 = lambda x: (x - mean_2) / std_2**2
elif simul_params['laws'] == 'norm-arctan':
    dpot_1 = lambda x: (x - mean_1) / std_1**2
    dpot_2 = lambda x: np.sin(x) / np.cos(x)**3 - 2*np.tan(x)
    #dpot_2 = lambda x: 2*np.tan(x) - (np.tan(x) - mean_1)*(1 + np.tan(x)**2) / std_1
elif simul_params['laws'] == 'norm-exp':
    dpot_1 = lambda x: (x - mean_1) / std_1**2
    dpot_2 = lambda x: (1 + (np.log(x) - mean_1)/(std_1**2)) / x
elif simul_params['laws'] == 'exp':
    potential_1 = lambda x: utils.exponential_potential(x, loc_1, scale_1)
    potential_2 = lambda x: utils.exponential_potential(x, loc_2, scale_2)

grad_potentials = [dpot_1, dpot_2]


h = simul_params['kernel_width']
norm2 = lambda x: x.dot(x)
kernel = lambda x, y: np.exp(-norm2(x-y) / h) / np.sqrt(h)
dkernel = lambda x, y: 2*(x-y)*np.exp(-norm2(x-y) / h) / (np.sqrt(h) * h)


def svgd_bary_gd(mesh, grad_potentials, kernel, grad_kernel, simul_params):
    def bary_iteration(L):
        L_new = np.copy(L)
        grads = np.zeros(L.shape)

        for alpha in range(simul_params['n_marginals']):

            for i in range(simul_params['n_samples']):
                gradk = utils.svgd_gradient(kernel, grad_kernel, grad_potentials[alpha], L[alpha], simul_params['n_samples'], i) / simul_params['n_samples']

                grad = \
                    simul_params['gd_step_size']*simul_params['regularization']*simul_params['bary_coords'][alpha] \
                    * gradk / simul_params['n_samples']
                
                dc = simul_params['gd_step_size']*utils.dc(L[:, i], simul_params['bary_coords'], convex=simul_params['g_conv'])[alpha]

                L_new[alpha, i] = + grad - dc + L[alpha, i]
                grads[alpha, i] = -grad + dc
        diffgrad = L_new - L
        
        return L_new, grads, diffgrad

    particle_trajectories = []

    initial_domain_start, initial_domain_end = simul_params['initial_domain'][0], simul_params['initial_domain'][1]
    initial_domain_length = initial_domain_end - initial_domain_start
    L = S.uniform.rvs(loc=initial_domain_start, scale=initial_domain_length, size=simul_params['n_marginals']*simul_params['n_samples']) \
        .reshape((simul_params['n_marginals'], simul_params['n_samples'], 1))
    L_init = np.copy(L)

    particle_trajectories.append(np.copy(L))

    plt.ion()
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.canvas.draw()

    for iteration in range(simul_params['n_iterations']):
        print('iter:', iteration)

        L, grads, diffgrad = bary_iteration(L)
        particle_trajectories.append(np.copy(L))

        if iteration > 0 and iteration % 20 == 0:
            xsort = np.argsort(L[0, :], axis=0)
            ysort = np.argsort(L[1, :], axis=0)
            if np.all(np.isclose(xsort, ysort)) and np.max(np.max(np.abs(grads))) < 1:
                simul_params['regularization'] *= 2
                simul_params['gd_step_size'] /= 2
                print('doubled alpha')

        if iteration % draw_freq == 0:
            axs[0].clear()
            axs[1].clear()

            axs[0].set_title('iteration {} -- coupling'.format(iteration))
            #axs[0].set_xlim(mesh.min(), mesh.max())
            #axs[0].set_ylim(mesh.min(), mesh.max())
            axs[0].grid()
            axs[0].scatter(L[0, :], L[1, :], marker='o', c='r', edgecolor='black')

            axs[1].set_title('physical view -- alpha={}, ss={}'.format(round(simul_params['regularization'], 2), round(simul_params['gd_step_size'], 5)))
            axs[1].grid()
            axs[1].hist(L[0, :], bins=50, density=True, color='r', alpha=0.3)
            axs[1].hist(L[1, :], bins=50, density=True, color='b', alpha=0.3)
            axs[1].scatter(L_init[0, :], np.zeros(L.shape[1]), c='black', marker='o', alpha=0.05)
            axs[1].scatter(L_init[1, :], np.zeros(L.shape[1]), c='black', marker='o', alpha=0.05)
            axs[1].scatter(L[0, :], diffgrad[0, :], c='r', marker='x', alpha=0.5)
            axs[1].scatter(L[0, :], np.zeros(L[0, :].size), c='r', marker='x')
            axs[1].scatter(L[0, :], grads[0, :], c='g', alpha=0.5, marker='o')
            axs[1].scatter(L[1, :], diffgrad[1, :], c='b', marker='x', alpha=0.5)
            axs[1].scatter(L[1, :], np.zeros(L[1, :].size), c='r', marker='x')
            axs[1].scatter(L[1, :], grads[1, :], c='b', alpha=0.5, marker='o')
            #axs[1].plot(mesh, marginal_1, c='black', alpha=0.3)
            #axs[1].plot(mesh, marginal_2, c='black', alpha=0.3)
            #axs[1].plot(mesh, bary, c='green', alpha=0.3)
            #axs[1].set_xlim(mesh.min(), mesh.max())
            if simul_params['laws'] == 'norm-arctan':
                axs[1].set_ylim(-1, 1)

            plt.pause(0.001)

    plt.ioff()
    plt.close()

    return L_init, L, np.array(particle_trajectories)

# ----
# | SIMULATION BEGINS
# ----

# execute simulation and estimate the density of the barycenter using a gaussian kernel estimator
init_samples, swarm, trajectories = svgd_bary_gd(mesh, grad_potentials, kernel, dkernel, simul_params)
init_samples = init_samples.reshape((-1,))
samples = np.copy(swarm)
pushforward = np.sum(swarm.T.dot(np.array(bary_coords)), axis=0)
swarm = pushforward
valid_samples = swarm[np.where(np.logical_and(swarm < mesh.max(), swarm > mesh.min()))[0]]
bary_density = S.gaussian_kde(valid_samples.reshape((-1,)))

print('n_valid_samples:', valid_samples.size)

if simul_params['laws'] in ['norm', 'norm-arctan', 'norm-exp']:
    fig, axs = plt.subplots(nrows=2, ncols=1)

    origin = lambda x: np.exp(-(x - mean_1)**2/(2*std_1**2)) / np.sqrt(2*np.pi*std_1**2)
    if simul_params['laws'] == 'norm-arctan':
        push = lambda t: np.arctan(t)
        pushforward = lambda t: (1 + np.tan(t)**2) * np.exp(-np.tan(t)**2/2) / np.sqrt(2*np.pi)
    elif simul_params['laws'] == 'norm-exp':
        push = lambda t: np.exp(t)
        pushforward = lambda t: np.exp(-(np.log(t)-mean_1)**2/(2*std_1**2)) / (t*np.sqrt(2*np.pi)*std_1)
    else:
        push = lambda t: t + mean_2 - mean_1
        pushforward = lambda t: np.exp(-(t-mean_2)**2/(2*std_2**2)) / np.sqrt(2*np.pi*std_2**2)

    axs[0].set_title('Samples and Density Estimate (Marginals in grey)')
    if simul_params['laws'] == 'norm-arctan':
        axs[0].set_xlim(-np.pi/2, np.pi/2) ##
    else:
        axs[0].set_xlim(np.min(np.min(trajectories)), np.max(np.max(trajectories)))
    axs[0].scatter(init_samples, np.zeros(init_samples.size), c='b', label='Initial Samples')
    axs[0].scatter(valid_samples, np.zeros(valid_samples.size), c='r', label='BARYGD Samples')
    axs[0].plot(mesh, bary_density.pdf(mesh), c='b', alpha=0.7, label='Estimated Density')
    axs[0].plot(mesh, origin(mesh), c='gray', alpha=0.3)
    axs[0].plot(mesh, pushforward(mesh), c='gray', alpha=0.3)
    axs[0].hist(valid_samples, density=True, color='r', alpha=0.1, bins='auto')

    # approximate true barycenter by sampling from the normal:
    normal_samples = S.norm.rvs(loc=mean_1, scale=std_1, size=10000)
    if simul_params['laws'] == 'norm-arctan':
        normal_samples = (1-l)*normal_samples + l*np.arctan(normal_samples)
    elif simul_params['laws'] == 'norm-exp':
        normal_samples = (1-l)*normal_samples + l*np.exp(normal_samples)
    else:
        normal_samples = (1-l)*normal_samples + l*(normal_samples + mean_2 - mean_1)
    bary_true_density = S.gaussian_kde(normal_samples)
    if simul_params['laws'] == 'norm-arctan':
        exp_label = r'$\mathrm{bar}(\mathscr{N}(0,1), \arctan {}_{\#} \mathscr{N}(0,1))$'
    else:
        exp_label = r'$\mathrm{bar}(\mathscr{N}(\mu_1,1), \mathscr{N}(\mu_2,1))$'
    axs[0].plot(mesh, bary_true_density.pdf(mesh), c='black', linewidth=2, alpha=0.9, label=exp_label)
    axs[0].grid()
    axs[0].legend()

    """
    axs[1].set_title('samples evolution')
    axs[1].grid()
    colors = ['black', 'grey', 'lightcoral', 'red', 'orangered', 'peru', 'goldenrod', \
        'olive', 'palegreen', 'turquoise', 'dodgerblue', 'blue', 'indigo', 'plum',
        'violet', 'magenta', 'orchid', 'hotpink', 'crimson', 'pink']
    the_chosen_ones = np.random.choice(range(simul_params['n_samples']), size=20)
    t = np.arange(trajectories.shape[0]).reshape((-1, 1))
    G_samples = np.sum([trajectories[:, j]*simul_params['bary_coords'][j] for j in range(simul_params['n_marginals'])], axis=0)

    for j, k in enumerate(the_chosen_ones):
        path = np.append(G_samples[:, k], t, axis=1)
        col = colors[j]
        axs[1].plot(path[:, 0], path[:, 1], '-', c=col, alpha=0.2, zorder=1)
        axs[1].scatter([path[0, 0]], [path[0, 1]], marker='o', edgecolor='black', c=col, alpha=0.5, zorder=5)
        axs[1].scatter([path[-1, 0]], [path[-1, 1]], marker='o', edgecolor='black', c=col, zorder=10)
    """

    axs[1].set_title(r'Evolution of $X_t$ (Initial samples in green, Transport map in gold)')

    valid_1 = np.where(np.logical_and(samples[0] < mesh.max(), samples[0] > mesh.min()))[0]
    valid_2 = np.where(np.logical_and(samples[1] < mesh.max(), samples[1] > mesh.min()))[0]
    valid = np.intersect1d(valid_1, valid_2)

    axs[1].plot(mesh, push(mesh), c='gold', alpha=0.7, zorder=50, linewidth=2)

    # ASSUMPTION: no point exploded
    axs[1].scatter(trajectories[0, 0], trajectories[0, 1], marker='o', c='g', edgecolor='black', alpha=0.3, zorder=5)
    for k in range(simul_params['n_samples']):
        path = trajectories[:, :, k].reshape(-1, 2)
        axs[1].plot(path[:, 0], path[:, 1], '-', c='black', markeredgecolor='black', alpha=0.2, zorder=1)
    axs[1].scatter(samples[0, valid], samples[1, valid], marker='o', c='r', edgecolor='black', zorder=10)


    if simul_params['laws'] == 'norm-arctan':
        axs[1].set_xlim(-np.pi/2, np.pi/2) ##
    else:
        #axs[1].set_xlim(np.min(np.min(trajectories))-2, np.max(np.max(trajectories)))
        axs[1].set_xlim(-6, 4)
        axs[1].set_ylim(-2, 10)
    #t = np.linspace(0.4, 1, 300) ##
    #axs[1, 0].plot(t, t**3, c='gray') ##
    #axs[1, 0].set_xlim(mesh.min(), mesh.max())
    #axs[1, 0].set_ylim(mesh.min(), mesh.max())
    axs[1].grid()

    fig.set_size_inches(8, 10)
    fig.savefig('./img/svgd_{}.png'.format(exp_name))
    plt.show()
