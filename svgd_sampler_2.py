import os
import seaborn as sea
import argparse
import pdb
import numpy as np
from progress.bar import Bar
from numpy.linalg import eigh, inv, norm, det
import scipy.stats as S
import autograd.numpy as anp
from sklearn.preprocessing import normalize
from autograd import grad, hessian
from autograd import elementwise_grad as egrad
import matplotlib.pyplot as plt
import subprocess
from utils import svgd_gradient, dc, norm2
import datetime
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from fat_stick import fat_stick, dfat_stick


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
bary_coords = np.array([0.4, 0.6])

simul_params = {
    'n_schemes': 1,
    'dimension':1,
    'n_eigenvalues': -1,
    'langevin_perturbation': False,
    'langevin_factor': 1,
    'g_conv': True,
    'n_iterations': 350,
    'initial_domain':  [(1.5, 1.6)],
    'fd_step_size': (mesh_x_max - mesh_x_min) / (n_mesh - 1),
    'tfd_step_size': 1e1,
    'gd_step_size':0.5e-3,
    'regularization': 1000,
    'n_samples': 50,
    'bary_coords': bary_coords,
    'n_marginals': len(bary_coords),
    'laws': '2d-sticks',
    'kernel_width': 1
}
if '2d' in simul_params['laws']: simul_params['dimension'] = 2

exp_name = 'bar_' + now_str + ''.join(['{}:{}_'.format(k[:4], simul_params[k]) for k in ['dimension', 'g_conv', 'n_iterations', 'regularization', 'gd_step_size', 'laws']])
print('NAME:', exp_name)
print('n_mesh: {}\t fd_step_size: {}'.format(n_mesh, simul_params['fd_step_size']))


# computing kernels for the marginals and their gradients
grad_potentials = []

if simul_params['laws'] == 'norm':
    mean_1, mean_2 = -2, 6
    std_1, std_2 = 1, 1
    grad_potentials.append(lambda x: (x - mean_1) / std_1**2)
    grad_potentials.append(lambda x: (x - mean_2) / std_2**2)
elif simul_params['laws'] == 'norm-arctan':
    grad_potentials.append(lambda x: (x - mean_1) / std_1**2)
    grad_potentials.append(lambda x: np.sin(x) / np.cos(x)**3 - 2*np.tan(x))
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
    grad_potentials.append(lambda x: istds[0].dot(x - means[0]))
    grad_potentials.append(lambda x: istds[1].dot(x - means[1]))
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


h = simul_params['kernel_width']
kernel = lambda x, y: np.exp(-norm2(x-y) / h) / np.sqrt(h)
dkernel = lambda x, y: 2*(x-y)*np.exp(-norm2(x-y) / h) / (np.sqrt(h) * h)


def svgd_bary_gd(mesh, grad_potentials, kernel, grad_kernel, simul_params):
    def bary_iteration(L):
        L_new = np.copy(L)
        grads = np.zeros(L.shape)

        for alpha in range(simul_params['n_marginals']):
            for i in range(simul_params['n_samples']):
                gradk = svgd_gradient(kernel, grad_kernel, grad_potentials[alpha], L[alpha], simul_params['n_samples'], i)
                grad = \
                    simul_params['gd_step_size']*simul_params['regularization']*simul_params['bary_coords'][alpha] \
                    * gradk / simul_params['n_samples']
                dcn = simul_params['gd_step_size']*dc(L[:, i], simul_params['bary_coords'], convex=simul_params['g_conv'])[alpha]

                L_new[alpha, i] = + grad - dcn + L[alpha, i]
                grads[alpha, i] = grad - dcn
        diffgrad = L_new - L
        
        return L_new, grads, diffgrad

    particle_trajectories = []

    if len(simul_params['initial_domain']) == 1:
        simul_params['initial_domain'] = [simul_params['initial_domain'][0] for _ in range(simul_params['n_marginals'])]

    L = np.zeros((simul_params['n_marginals'], simul_params['n_samples'], simul_params['dimension']))
    for k in range(simul_params['n_marginals']):
        dom_min, dom_max = simul_params['initial_domain'][k][0], simul_params['initial_domain'][k][1]
        dom_size = dom_max - dom_min
        Lk = S.uniform.rvs(loc=dom_min, scale=dom_size, size=simul_params['n_samples']*simul_params['dimension'])\
            .reshape((1, simul_params['n_samples'], simul_params['dimension']))
        L[k] = Lk
    L_init = np.copy(L)

    particle_trajectories.append(np.copy(L))

    plt.ion()
    if simul_params['dimension'] == 1:
        if simul_params['n_marginals'] == 2:
            fig, axs = plt.subplots(nrows=1, ncols=2)
        else:
            fig = plt.figure(figsize=plt.figaspect(2))
            ax0 = fig.add_subplot(2, 1, 1, projection='3d')
            ax1 = fig.add_subplot(2, 1, 2)
            axs = [ax0, ax1]
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.canvas.draw()
    cols = ['red', 'orange', 'purple']
    marks = ['o', '^', 'x']

    if simul_params['laws'] == '2d-sticks':
        x0 = np.linspace(-1, 4, 300)
        y0 = np.linspace(-1, 4, 300)
        X0, Y0 = np.meshgrid(x0, y0)

    for iteration in range(simul_params['n_iterations']):
        print('iter:', iteration)

        L, grads, diffgrad = bary_iteration(L)
        particle_trajectories.append(np.copy(L))

        if iteration > 0 and iteration % 20 == 0:
            xsort = np.argsort(L[0, :], axis=0)
            ysort = np.argsort(L[1, :], axis=0)
            if simul_params['laws'].startswith('3'):
                zsort = np.argsort(L[2, :], axis=0)

            if (simul_params['n_marginals'] == 2 and np.all(np.isclose(xsort, ysort)) and np.max(np.max(np.abs(grads))) < 1) \
                or (simul_params['n_marginals'] == 3 and np.all(np.isclose(xsort, ysort)) and np.all(np.isclose(xsort, zsort)) \
                    and np.max(np.max(np.abs(grads))) < 1):
                simul_params['regularization'] *= 2
                simul_params['gd_step_size'] /= 2
                print('alpha doubled: (alpha, gd_step_size) = ({}, {})'\
                    .format(simul_params['regularization'], simul_params['gd_step_size']))
            if simul_params['laws'] == '2d-sticks' and iteration > 29:
                simul_params['regularization'] *= 2
                simul_params['gd_step_size'] /= 2 

        if iteration % draw_freq == 0:
            if simul_params['dimension'] == 1:
                axs[0].clear()
                axs[1].clear()

                axs[0].set_title('iteration {} -- coupling'.format(iteration))
                #axs[0].set_xlim(mesh.min(), mesh.max())
                axs[0].grid()
                if simul_params['n_marginals'] == 2:
                    axs[0].scatter(L[0, :], L[1, :], marker='o', c='r', edgecolor='black')
                elif simul_params['n_marginals'] == 3:
                    axs[0].scatter(L[0, :], L[1, :], L[2, :], marker='o', c='r', edgecolor='black')


                axs[1].set_title('physical view -- alpha={}, ss={}'.format(round(simul_params['regularization'], 2), round(simul_params['gd_step_size'], 5)))
                axs[1].grid()
                for k in range(simul_params['n_marginals']):
                    axs[1].hist(L[k, :], bins=50, density=True, color=cols[k], alpha=0.3)
                    axs[1].scatter(L_init[k, :], np.zeros(L.shape[1]), c='black', marker='o', alpha=0.05)
                    #axs[1].scatter(L[k, :], diffgrad[k, :], c=cols[k], marker=marks[k], alpha=0.5)
                    axs[1].scatter(L[k, :], np.zeros(L[k, :].size), c=cols[k], marker=marks[k])
                    axs[1].scatter(L[k, :], grads[k, :], c=cols[k], alpha=0.5, marker=marks[k])
                #axs[1].set_xlim(mesh.min(), mesh.max())
                if simul_params['laws'] == 'norm-arctan':
                    axs[1].set_ylim(-1, 1)
            else:
                ax.clear()
                ax.set_title('iteration {} -- (al,sg)=({},{})'.format(iteration, simul_params['regularization'], round(simul_params['gd_step_size'], 4)))
                #ax.set_xlim(-6, 6)
                #ax.set_ylim(-6, 6)
                ax.grid()

                for k in range(simul_params['n_marginals']):
                    samples = np.sum(np.array([L[k]*bary_coords[k] for k in range(simul_params['n_marginals'])]), axis=0)
                    sea.kdeplot(x=samples[:, 0], y=samples[:,1], fill=True, color='gray', alpha=0.4)
                    sea.kdeplot(x=L[k, :, 0], y=L[k, :, 1], fill=True, color=cols[k], alpha=0.3)
                    ax.scatter(L[k, :, 0], L[k, :, 1], c=cols[k], marker=marks[k], edgecolor='black')
                    ax.scatter(samples[:, 0], samples[:, 1], c='gray', marker='x', edgecolor='black')

                    if simul_params['laws'] == '2d-sticks':
                        plt.contour(X0, Y0, fat_stick(X0, Y0, A[k], B[k], 5, eta=eta), cmap='Dark2', levels=50, alpha=0.1)

            plt.pause(0.001)

    plt.ioff()
    plt.close()

    return L_init, L, np.array(particle_trajectories)

# ----
# | SIMULATION BEGINS
# ----

# execute simulation and estimate the density of the barycenter using a gaussian kernel estimator
init_samples, swarm, trajectories = svgd_bary_gd(mesh, grad_potentials, kernel, dkernel, simul_params)
if simul_params['dimension'] == 1:
    init_samples = np.sum(init_samples.T.dot(bary_coords), axis=0).reshape((-1,))
    pushforward = np.sum(swarm.T.dot(np.array(bary_coords)), axis=0)
    samples = np.copy(swarm)
    swarm = pushforward
    valid_samples = swarm[np.where(np.logical_and(swarm < mesh.max(), swarm > mesh.min()))[0]]
    bary_density = S.gaussian_kde(valid_samples.reshape((-1,)))

    print('n_valid_samples:', valid_samples.size)
else:
    init_samples = np.sum(np.array([init_samples[k]*bary_coords[k] for k in range(simul_params['n_marginals'])]), axis=0)
    pushforward = np.sum(np.array([swarm[k]*bary_coords[k] for k in range(simul_params['n_marginals'])]), axis=0)


if simul_params['laws'] in ['2d-norm', '2d-3-norm', '2d-sticks']:
    fig = plt.figure(figsize=(12, 8))
    #plt.xlim(-12, 12)
    #plt.ylim(-1, 12)

    densities = []
    for k in range(simul_params['n_marginals']):
        if simul_params['laws'] in ['2d-norm', '2d-3-norm']:
            densities.append(lambda x, y, k=k: np.exp(-0.5*(istds[k][0,0]*(x-means[k][0])**2 + 2*istds[k][0,1]*(x-means[k][0])*(y-means[k][1]) + istds[k][1,1]*(y-means[k][1])**2)) / (2*np.pi*det(stds[k])))

    plt.title('Samples and Density Estimate -- Algorithm: BARYGD-SVGD')
    plt.grid()
    plt.scatter(init_samples[:, 0], init_samples[:, 1], c='lime', marker='^', edgecolor='black', label='Initial Samples', zorder=5, alpha=0.7)
    plt.scatter(pushforward[:, 0], pushforward[:, 1], c='r', marker='o', edgecolor='black', label='BARYGD Samples', zorder=10)
    sea.kdeplot(x=pushforward[:, 0], y=pushforward[:, 1], fill=True, color='r', alpha=0.8, shadelowest=True, bw_method='silverman', bw_adjust=1.6)

    marginal_samples = []
    X, Y = np.meshgrid(mesh, mesh) if simul_params['laws'] != '2d-sticks' else np.meshgrid(np.linspace(-1, 4, 300), np.linspace(-1, 3, 300))
    cols = ['winter', 'hot', 'viridis', 'coolwarm', 'autumn', 'YlOrRd']
    label_patches = []
    for k in range(simul_params['n_marginals']):
        if simul_params['laws'] != '2d-sticks':
            samples = np.random.multivariate_normal(means[k], stds[k], size=10000)
            marginal_samples.append(samples)
        sea.kdeplot(x=swarm[k, :, 0], y=swarm[k, :, 1], cmap=cols[k], fill=True, alpha=0.3)
        label_patches.append(mpatches.Patch(color=sea.color_palette(cols[k])[2], label='Marginal {} Estimate'.format(k)))
        label_patches.append(mpatches.Patch(color=sea.color_palette(cols[k+2])[2], label='Marginal {} True'.format(k)))
        if simul_params['laws'] != '2d-sticks':
            dens = densities[k](X, Y)
        else:
            dens = fat_stick(X, Y, A[k], B[k], 10, eta=eta)
        plt.contour(X, Y, dens, cmap=cols[k+2])
    if simul_params['laws'] != '2d-sticks':
        marginal_samples = np.array(marginal_samples)
        true_bary_samples = np.sum(np.array([marginal_samples[k]*bary_coords[k] for k in range(simul_params['n_marginals'])]), axis=0)
        sea.kdeplot(x=true_bary_samples[:, 0], y=true_bary_samples[:, 1], fill=False, color='black', alpha=1)
        label_patches.append(mpatches.Patch(color='black', label='Barycenter Density'))
    label_patches.append(mpatches.Patch(color='red', label='Barycenter Density Estimate'))

    bt = []
    for t in range(trajectories.shape[0]):
        bt.append(np.sum(np.array([trajectories[t, k]*bary_coords[k] for k in range(simul_params['n_marginals'])]), axis=0))
    bt = np.array(bt)
    for k in range(simul_params['n_samples']):
        plt.plot(bt[:, k, 0], bt[:, k, 1], '-', c='black', alpha=0.1, zorder=1)

    ax = plt.gca()
    label_handles, _ = ax.get_legend_handles_labels()
    label_handles = label_handles + label_patches
    ax.legend(handles = label_handles)
    fig.savefig('./img/svgd_{}.png'.format(exp_name))
    plt.show()


if simul_params['laws'].startswith('3'):
    fig = plt.figure(figsize=plt.figaspect(2))
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2, projection='3d')
    axs = [ax0, ax1]

    if simul_params['laws'] == '3-norm':
        origin = lambda x: np.exp(-0.5*(x-means[0])**2/stds[0]**2) / np.sqrt(2*np.pi*stds[0]**2)
        T = lambda x,m,s: (x - means[0])*s/stds[0] + m
        Tinv = lambda x,m,s: (x - m)*stds[0]/s + means[0]
        push = lambda x: sum([bary_coords[k]*T(x, means[k], stds[k]) for k in range(simul_params['n_marginals'])])
    elif simul_params['laws'] == '3-norm-arctan-exp':
        origin = lambda x: np.exp(-0.5*(x - mean) / std**2) / np.sqrt(2*np.pi*std**2)
        densities = []; 
        densities.append(lambda x: x); 
        densities.append(lambda x: (1+np.tan(x)**2)*origin(np.tan(x)))
        densities.append(lambda x: origin(np.log(x)) / x)
        T = []
        T.append(lambda x: x)
        T.append(lambda x: np.arctan(x))
        T.append(lambda x: np.exp(x))
        push = lambda x: sum([bary_coords[k]*T[k](x) for k in range(simul_params['n_marginals'])])
    elif simul_params['laws'] == '3-norm-arctan-norm':
        origin = lambda x: np.exp(-0.5*(x - means[0])**2 / stds[0]**2) / np.sqrt(2*np.pi*stds[0]**2)
        domains = [mesh, np.linspace(-np.pi/2+1e-3, np.pi/2-1e-3, 300), mesh]
        densities = []
        densities.append(origin)
        densities.append(lambda x: (1+np.tan(x)**2)*origin(np.tan(x)))
        densities.append(lambda x: np.exp(-(x-means[1])**2/(2*stds[1]**2)) / np.sqrt(2*np.pi*stds[1]**2))
        T = []
        T.append(lambda x: x)
        T.append(lambda x: np.arctan(x))
        T.append(lambda x: (x - means[0])*stds[1]/stds[0] + means[1])
        push = lambda x: sum([bary_coords[k]*T[k](x) for k in range(simul_params['n_marginals'])])


    axs[0].set_title('Samples and Density Estimate (Marginals in grey) -- Algorithm: BARYGD-SVGD')
    axs[0].scatter(init_samples[::20], np.zeros(init_samples[::20].size)-0.005, c='lime', marker='^', edgecolor='black', label='Initial Samples', zorder=5, alpha=0.7)
    axs[0].scatter(init_samples[::21], np.zeros(init_samples[::21].size)+0.005, c='lime', marker='^', edgecolor='black', zorder=5, alpha=0.7)
    axs[0].scatter(valid_samples[::2], np.zeros(valid_samples[::2].size), c='r', marker='o', edgecolor='black', label='BARYGD Samples')
    axs[0].plot(mesh, bary_density.pdf(mesh), c='b', alpha=0.7, label='Estimated Barycenter Density')
    for k in range(simul_params['n_marginals']):
        if simul_params['laws'] == '3-norm':
            axs[0].plot(mesh, (stds[0]/stds[k])*origin(Tinv(mesh, means[k], stds[k])), c='gray', alpha=0.3)
        elif simul_params['laws'] == '3-norm-arctan-exp':
            axs[0].plot(mesh, densities[k](mesh), c='gray', alpha=0.3)
        elif simul_params['laws'] == '3-norm-arctan-norm':
            axs[0].plot(domains[k], densities[k](domains[k]), c='gray', alpha=0.3)
        print('ploted marginal', k)
    axs[0].hist(valid_samples, density=True, color='r', alpha=0.1, bins='auto')
    if simul_params['laws'] == '3-norm':
        true_bary_samples = S.norm.rvs(loc=means[0], scale=stds[0], size=10000)
    elif simul_params['laws'] == '3-norm-arctan-exp':
        true_bary_samples = S.norm.rvs(loc=mean, scale=std, size=10000)
    elif simul_params['laws'] == '3-norm-arctan-norm':
        true_bary_samples = S.norm.rvs(loc=means[0], scale=stds[0], size=10000)
    true_bary_samples = push(true_bary_samples)
    bary_true_density = S.gaussian_kde(true_bary_samples)
    axs[0].plot(mesh, bary_true_density.pdf(mesh), c='black', linewidth=2, alpha=0.9, label='Barycenter Density')
    if simul_params['laws'] in ['3-norm-arctan-exp', '3-norm-arctan-norm']:
        #axs[0].set_ylim(0, 1)
        if simul_params['laws'] == '3-norm-arctan-exp': 
            axs[0].set_xlim(-np.pi, np.pi)
    axs[0].grid()
    axs[0].legend()

    axs[1].set_title(r'Evolution of $X_t$ (Initial samples in green, Transport map in gold)')
    axs[1].scatter(trajectories[0, 0], trajectories[0, 1], trajectories[0, 2], marker='o', c='g', edgecolor='black', alpha=0.3, zorder=5)
    for k in range(simul_params['n_samples']):
        path = trajectories[:, :, k].reshape(-1, 3)
        axs[1].plot(path[:, 0], path[:, 1], path[:, 2], '-', c='black', markeredgecolor='black', alpha=0.2, zorder=1)
    axs[1].scatter(samples[0, :], samples[1, :], samples[2, :], marker='o', c='r', edgecolor='black', zorder=10)
    def T_bar(x):
        if simul_params['laws'] == '3-norm':
            return [T(x, means[0], stds[0]), T(x, means[1], stds[1]), T(x, means[2], stds[2])]
        else:
            return [T[k](x) for k in range(simul_params['n_marginals'])]
    coupling = T_bar(mesh)
    axs[1].plot(coupling[0], coupling[1], coupling[2], linewidth=2, c='gold')
    axs[1].grid()
    axs[1].set_xlim(np.min(np.min(trajectories[:, 0])), np.max(np.max(trajectories[:, 0])))
    axs[1].set_ylim(np.min(np.min(trajectories[:, 1])), np.max(np.max(trajectories[:, 1])))
    axs[1].set_zlim(np.min(np.min(trajectories[:, 2])), np.max(np.max(trajectories[:, 2])))

    fig.set_size_inches(8, 10)
    fig.savefig('./img/svgd_{}.png'.format(exp_name))
    plt.show()


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
    l = bary_coords[0]
    normal_samples = S.norm.rvs(loc=mean_1, scale=std_1, size=10000)
    if simul_params['laws'] == 'norm-arctan':
        normal_samples = l*normal_samples + (1-l)*np.arctan(normal_samples)
    elif simul_params['laws'] == 'norm-exp':
        normal_samples = l*normal_samples + (1-l)*np.exp(normal_samples)
    else:
        normal_samples = l*normal_samples + (1-l)*(normal_samples - mean_1 + mean_2)
    bary_true_density = S.gaussian_kde(normal_samples)
    if simul_params['laws'] == 'norm-arctan':
        exp_label = r'$\mathrm{bar}(\mathscr{N}(0,1), \arctan {}_{\#} \mathscr{N}(0,1))$'
    else:
        exp_label = r'$\mathrm{bar}(\mathscr{N}(\mu_1,1), \mathscr{N}(\mu_2,1))$'
    axs[0].plot(mesh, bary_true_density.pdf(mesh), c='black', linewidth=2, alpha=0.9, label=exp_label)
    axs[0].grid()
    axs[0].legend()

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
