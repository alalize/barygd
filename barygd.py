import os
import sys
import shutil
from scipy.integrate import quad
from scipy.stats.kde import gaussian_kde
import ot
import argparse
import time
import seaborn as sea
import argparse
import pdb
import numpy as np
import json
from progress.bar import Bar
from numpy.linalg import eigh, inv, norm, det
import scipy.stats as S
from scipy.stats import entropy as KL
from scipy.linalg import sqrtm
import autograd.numpy as anp
from sklearn.preprocessing import normalize
from autograd import grad, hessian
from autograd import elementwise_grad as egrad
import matplotlib
if os.uname().nodename != 'azra':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import utils
from utils import lawgd_gradient, svgd_gradient, dc, svgd_adaptive_gradient, laws_2norm_family, intc, norm2
import datetime
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FFMpegWriter
from fat_stick import fat_stick, dfat_stick
from plot import plot_init, plot_intermediate, plot_free

from kernel import SVGDKernel
from solver import Solver

from mixture_bary_lp import sample_cdf


if __name__ == '__main__':
    bool2int = lambda b: 1 if b else 0

    np.random.seed(0)
    now_str = str(datetime.datetime.now()).replace(' ', '_')
    draw_freq = 10
    w2_freq = 2
    threading = False

    n_mesh = 256
    mesh_x_min, mesh_x_max = -10, 10
    mesh = np.linspace(mesh_x_min, mesh_x_max, num=n_mesh)

    iss0 = 1e-2
    glob_dim = 8
    bary_gaussian_samples = 300
    bary_coords = np.array([0.5, 0.5])

    # mode \in ['run', 'dependance-regularization', 'dependance-dimension', dependance-n-marginals', 'dependance-samples']
    # also 'w2-iterations' with SVGD & LAWGD curves for all considered laws

    simul_params = {
        'mode': 'dependance-iterations',
        'alpha-freq': 5,
        'frozen-step': False,
        'print-gradients': False,
        'check-converged': True,
        'convergence-threshold': 1e-4,
        'n_eigen': 50,
        'g_conv': False,
        'coulomb': True,

        'n_iterations': 200,

        'initial_domain': [(-0.1, 0.1)],
        'initial-domain': (8, 10),
        'fd_step_size': (mesh_x_max - mesh_x_min) / (n_mesh - 1),
        'gd_step_size': 1e-4,
        'initial-gd-step-size': iss0,
        'iss0': iss0,
        'min-bandwidth': 1,

        'adaptive-step-size': True,
        'adaptive-step-size-method': 'adagrad', # momentum or adagrad

        'momentum': 1,
        'adaptive-kernel': False,
        'kernel-order': 2,

        'regularization': 1000,

        'samples-per-marginal': 50,
        'bary_coords': bary_coords.tolist(),
        'n_marginals': len(bary_coords),

        'laws': 'norms',

        'orthogonal-normals': True,
        'compute-w2': False,
        'track-w2': False,
        'track-mmd': False,
        'estimate-w2': False,
        'track-integral': True,
        'equal-coupling-start': False,

        'keep-regularization-fixed': False,

        'smooth-regularization-augment': False,
        'num-augment-regul': 50,
        'final-regularization': 1e9,

        'plot': True, 

        'film': True,
        'many-films': True,
        'film_name': 'coupling-optimization',
        'algorithm': 'svgd',
        'rotate-view': False,
        'film_desc': {
            'title': 'BARYGD Sampling',
            'artist': 'Majabaut',
            'comment': 'LAWGD variant'
        }
    }
    simul_params['dimension'] = 2 if '2d' in simul_params['laws'] else 1
    if simul_params['laws'] == 'd-norms':
        simul_params['dimension'] = glob_dim
    if simul_params['track-mmd']:
        simul_params['track-w2'] = True
    if simul_params['track-w2'] and not simul_params['track-mmd']:
        simul_params['compute-w2'] = True


    parser = argparse.ArgumentParser()
    parser.add_argument('-iss', type=float, help='Initial step size for gradient descent.')
    parser.add_argument('-penal', type=float, help='Penalization strength for BARYGD.')
    parser.add_argument('--auto-iss', action='store_true', help='If provided, initial-gd-step-size is computed from penalization.')
    args = parser.parse_args()

    if args.iss:
        simul_params['initial-gd-step-size'] = args.iss
    if args.penal:
        simul_params['regularization'] = args.penal
    if args.auto_iss:
        if args.iss:
            print('(Warning)  You should not pass argument -iss when --auto-iss is provided. Value -iss discarded.')
        simul_params['initial-gd-step-size'] = 1 / (simul_params['regularization']*1e2)
    simul_params['gd_step_size'] = simul_params['initial-gd-step-size']


    file_count = len(os.listdir('../img'))
    print('File count: ', file_count)
    exp_name = '{}_{}bar_'.format(file_count+1, 'c' if simul_params['coulomb'] else '') + now_str + ''.join(['_{}:{}_'.format(k[:4], simul_params[k]) for k in ['dimension', 'g_conv', 'n_iterations', 'regularization', 'gd_step_size', 'laws']])
    print('Name:', exp_name)
    print('Mesh size: {:>6}   Fin. Diff. step: {:>10e}'.format(n_mesh, simul_params['fd_step_size']))


    print('Creating experiment directory...')
    os.makedirs('../img/{}'.format(exp_name), exist_ok=True)
    print('Creating experiment directory... [Ok] (Might already exist.)')

    print('Logging simulation parameters...')
    with open('../img/{}/parameters.log'.format(exp_name), 'w') as f:
        f.write(json.dumps(simul_params, sort_keys=True, indent=4))
        f.write('\r\n')
    print('Logging simulation parameters... [Ok]')

    if simul_params['algorithm'] == 'lawgd':
        from lawgd_context import get_context
        cmesh = np.copy(mesh)
        if simul_params['dimension'] == 2:
            cmesh = np.array([(mesh[i], mesh[-1-j]) for j in range(n_mesh) for i in range(n_mesh)])
        gd_context, potentials, grad_potentials, lap_potentials, means, stds, istds, eta = get_context(cmesh, n_mesh, simul_params)
    elif simul_params['algorithm'] == 'svgd':
        from svgd_context import get_context
        gd_context, grad_potentials, means, stds, istds, eta = get_context(simul_params)
    else:
        raise ValueError('Unknown algorithm for marginals.')

    print('\nSimulation parameters:')
    for k in sorted(simul_params.keys()):
        print('\t{}: {}'.format(k, simul_params[k]))

    if simul_params['dimension'] == 1 and simul_params['compute-w2'] or simul_params['estimate-w2']:
        print('\nComputing distance matrix...')
        ot_distances = np.array([[(mesh[i] - mesh[j])**2 for i in range(mesh.size)] for j in range(mesh.size)])
        print('Computing distance matrix... [Ok]')

    if simul_params['laws'] not in ['norm-mixture']:
        simul_params['estimate-w2'] = False
    
    w2_marginals = None
    if simul_params['estimate-w2']:
        print('\nComputing Wasserstein distance between marginals...')

        if simul_params['laws'] == 'norm-mixture':
            gaussian_density = lambda x: np.exp(-0.5*(x - means[0])**2 / stds[0]**2) / (stds[0]*np.sqrt(2*np.pi))
            mixture_density = np.vectorize(utils.gaussian_mixture(means[-1], means[1], stds[1]))
            w2_marginals = np.sqrt(ot.emd2(gaussian_density(mesh), mixture_density(mesh), ot_distances))


# ----
# |
# | SIMULATION LOOP
# |
# ----


def barygd(gdc, simul_params, run=0, bary_density=None, push=None, w2_with_samples=True, w2_marginals=None):
    particle_trajectories = []
    w2_dists = []
    w2_iters = []
    intc_values = []

    if len(simul_params['initial_domain']) == 1:
        simul_params['initial_domain'] = [simul_params['initial_domain'][0] for _ in range(simul_params['n_marginals'])]

    #L = np.zeros((simul_params['n_marginals'], simul_params['samples-per-marginal'], simul_params['dimension']))
    #if simul_params['equal-coupling-start']:
    #    dim = simul_params['dimension']
    #    dom_min, dom_max = simul_params['initial_domain'][0][0], simul_params['initial_domain'][0][1]
    #    dom_size = dom_max - dom_min
    #    X0 = S.uniform.rvs(loc=dom_min, scale=dom_size, size=simul_params['dimension'])
    #    for k in range(simul_params['n_marginals']):
    #        for ell in range(simul_params['samples-per-marginal']):
    #            L[k, ell] = X0 + 1e-4*np.random.multivariate_normal(np.zeros(dim), np.eye(dim))

    #else:
    #    for k in range(simul_params['n_marginals']):
    #        dom_min, dom_max = simul_params['initial_domain'][k][0], simul_params['initial_domain'][k][1]
    #        dom_size = dom_max - dom_min
    #        Lk = S.uniform.rvs(loc=dom_min, scale=dom_size, size=simul_params['samples-per-marginal']*simul_params['dimension'])\
    #            .reshape((1, simul_params['samples-per-marginal'], simul_params['dimension']))
    #        L[k] = Lk
    #L_init = np.copy(L)

    #particle_trajectories.append(np.copy(L))

    if simul_params['plot']:
        context = plot_init(ion=True, simul_params=simul_params)
        if simul_params['film']:
            file_name = simul_params['film_name']
            if simul_params['many-films']:
                file_name = file_name + str(run)
            context['writer'].setup(fig=context['fig'], outfile='../img/{}/{}.mp4'.format(exp_name, file_name), dpi=100)

    bar = Bar('BARYGD:', max=simul_params['n_iterations'], suffix='ETA %(eta)d sec, %(index)d/%(max)d - %(percent)d %%')
    
    if simul_params['smooth-regularization-augment']:
        regul_augment_freq = simul_params['n_iterations'] // simul_params['num-augment-regul']
        regul_increment = simul_params['final-regularization'] / simul_params['num-augment-regul']
        print('(Run {})  Regul. Augment Frequency:  {:>10}  Regul. Increment:  {:>10}'.format(run, regul_augment_freq, regul_increment))
    
    svgd_kernel = SVGDKernel(potential_grads=gdc['dV'])
    solver = Solver(simul_params['n_marginals'], simul_params['samples-per-marginal'], simul_params['dimension'],\
        simul_params['initial-gd-step-size'], bary_coords, svgd_kernel, simul_params['regularization'], \
            adaptive_step_size=False, threading=threading, variable_alpha=not simul_params['keep-regularization-fixed'],
            alpha_freq=simul_params['alpha-freq'], frozen_step=simul_params['frozen-step'],
            coulomb=simul_params['coulomb'])
    solver.initialize_coupling('uniform', simul_params['initial-domain'])

    if simul_params['track-mmd']:
        other_solver = Solver(
            simul_params['n_marginals'],
            simul_params['samples-per-marginal'],
            simul_params['dimension'],
            simul_params['initial-gd-step-size'],
            bary_coords,
            svgd_kernel,
            simul_params['regularization'],
            adaptive_step_size=False,
            threading=threading,
            variable_alpha=not simul_params['keep-regularization-fixed'],
            alpha_freq=simul_params['alpha-freq'],
            frozen_step=simul_params['frozen-step']
        )
        other_solver.initialize_coupling('uniform', (-5, 5))

    L_init = np.copy(solver.coupling)
    particle_trajectories.append(L_init)

    if w2_marginals is not None:
        w2_to_marginals_estimates = []

    for iteration in range(simul_params['n_iterations']):
        bar.next()
        
        solver.update()
        L = solver.coupling
        grads = L - particle_trajectories[-1]
        particle_trajectories.append(np.copy(L))

        if simul_params['track-mmd']:
            other_solver.update()

        if iteration == simul_params['n_iterations']-1:
            print('\n(Iter. final)  Solver step: {}'.format(solver.gd_step))

        if iteration > 0 and iteration % w2_freq == 0 and simul_params['track-w2']:
            pf = np.sum(np.array([bary_coords[i]*L[i, :] for i in range(simul_params['n_marginals'])]), axis=0)

            if not simul_params['track-mmd']:
                if simul_params['dimension'] == 1:
                    valid = pf[np.where(np.logical_and(pf < mesh.max(), pf > mesh.min()))[0]]

                    if not w2_with_samples:
                        current_density = S.gaussian_kde(valid.reshape((-1,)))
                        w2_dist = np.sqrt(ot.emd2(current_density(mesh), bary_density(mesh), ot_distances))
                    else:
                        valid = np.squeeze(valid)
                        num_valid = len(valid)
                        valid_0 = np.copy(valid)
                        bary_density_ = np.squeeze(bary_density)

                        for _ in range(len(bary_density_) // num_valid):
                            perturbation = np.random.normal(loc=0, scale=1e-2, size=num_valid)
                            perturbed = valid_0 + perturbation
                            valid = np.hstack([valid, perturbed])

                        ot_dists = np.array([[(valid[i] - bary_density_[j])**2 for j in range(len(bary_density_))] for j in range(len(valid))])
                        valid = np.ones(len(valid)) / len(valid)
                        bary_density_ = np.ones(len(bary_density_)) / len(bary_density_)
                        w2_dist = np.sqrt(ot.emd2(valid, bary_density_, ot_dists))
                else:
                    valid = None
                    dim = simul_params['dimension']
                    N, M = simul_params['samples-per-marginal'], 500
                    pf_perturbed = np.zeros((N + M, dim), order='C')
                    for k in range(N+M):
                        pf_perturbed[k] = pf[k % N] + 1e-4*np.random.multivariate_normal(np.zeros(dim), np.eye(dim))
                    pf_perturbed = np.asarray(pf_perturbed, order='C')
                    dist_matrix = ot.dist(pf_perturbed, bary_density) # in higher dim., we are already given samples at call
                    a, b = np.ones((N+M,))/(N+M), np.ones((bary_density.shape[0],))/bary_density.shape[0]
                    w2_dist = np.sqrt(ot.emd2(a, b, dist_matrix, numItermax=1e6))
            else:
                other_pf = np.sum(np.array([bary_coords[i]*other_solver.coupling[i, :] for i in range(simul_params['n_marginals'])]), axis=0)
                half_bary_samples = len(bary_density) // 2
                w2_dist = utils.mmd(pf, other_pf, bary_density[:half_bary_samples], bary_density[half_bary_samples:])

            w2_iters.append(iteration+1)
            w2_dists.append(w2_dist)
            if simul_params['track-mmd']:
                print('\n(Iter. {})  MMD: {}'.format(iteration, w2_dist))
            else:
                suffix = '' if simul_params['track-integral'] else '\r\n'
                print('\r\n(Iter. {}) Valid {}, W2 distance: {:e}{}'.format(\
                    iteration, valid.size if valid is not None else '', w2_dist, suffix))

        if iteration > 0 and iteration % w2_freq == 0 and simul_params['track-integral']:
            intval = intc(L, bary_coords)

            prefix = '\t' if simul_params['track-w2'] else '(Iter. {})'.format(iteration)
            print('\n{} Integral Cost: {:e}'.format(prefix, intval))
            intc_values.append(intval)

        if iteration > 0 and iteration % draw_freq == 0:
            print('\n(Iter. {})  Alpha:{}  Step size: {}'.format(iteration, solver.alpha, solver.gd_step))

        if iteration > 0 and iteration % w2_freq == 0 and w2_marginals is not None:
            intval = intc(L, bary_coords)
            w2_coupling_estimate = 2 * np.sqrt(intval)
            w2_diff = np.abs(w2_coupling_estimate - w2_marginals)

            w2_to_marginals_estimates.append(w2_diff)
            print('\r\n(Iter. {})  W2(marginals) estimation error: {}  (POT: {})'.format(iteration, w2_diff, w2_marginals))

        if iteration > 0 and iteration % draw_freq == 0 or iteration == simul_params['n_iterations'] - 1:
            print('(Iter. {})  Extreme positions: ({}, {})'.format(iteration, np.min(L), np.max(L)))

            if simul_params['laws'] == 'd-norms':
                empirical_bary_samples = np.sum(np.array([bary_coords[i]*L[i, :] for i in range(simul_params['n_marginals'])]), axis=0)
                empirical_bary_mean = np.sum(empirical_bary_samples, axis=0) / simul_params['samples-per-marginal']
                empirical_bary_cov = np.sum([np.outer(empirical_bary_samples[i] - empirical_bary_mean, empirical_bary_samples[i] - empirical_bary_mean) for i in range(simul_params['samples-per-marginal'])], axis=0) / (simul_params['samples-per-marginal'] - 1)

                delta_mean = np.sqrt(utils.norm2(empirical_bary_mean - glob_bary_mean))
                delta_cov = empirical_bary_cov - glob_bary_cov
                delta_cov = np.sqrt(np.trace(delta_cov.T.dot(delta_cov)))

                print('L2(Emp. mean, mean): {}  L2(Emp. cov, cov): {}'.format(delta_mean, delta_cov))


        #if not simul_params['keep-regularization-fixed']:
        #    if simul_params['algorithm'] == 'svgd':
        #        xsort = np.argsort(L[0, :], axis=0)
        #        ysort = np.argsort(L[1, :], axis=0)
        #        if simul_params['laws'].startswith('3'):
        #            zsort = np.argsort(L[2, :], axis=0)

        #        if (simul_params['n_marginals'] == 2 and np.all(np.isclose(xsort, ysort)) and np.max(np.max(np.abs(grads))) < 1) \
        #            or (simul_params['n_marginals'] == 3 and np.all(np.isclose(xsort, ysort)) and np.all(np.isclose(xsort, zsort)) \
        #                and np.max(np.max(np.abs(grads))) < 1):
        #            solver.alpha *= 2
        #            solver.gd_step /= 2

        #            print('alpha doubled: (alpha, gd_step_size) = ({}, {})'.format(round(solver.alpha, 3), round(solver.gd_step, 3)))
        #        if simul_params['laws'] == '2d-sticks' and iteration > 29:
        #            simul_params['regularization'] *= 2
        #            simul_params['gd_step_size'] /= 2 
        #    else:
        #        if iteration > 0 and iteration%1000 == 0:
        #            simul_params['regularization'] *= 2
        
        if simul_params['smooth-regularization-augment']:
            if iteration > 0 and iteration % regul_augment_freq == 0:
                simul_params['regularization'] += regul_increment
                simul_params['initial-gd-step-size'] = 1/(1e2*simul_params['regularization'])
                print('\r\nRegularization augment... [Regul.: {:>10}  Step:  {:>10}]'.format(simul_params['regularization'], simul_params['initial-gd-step-size']))

        if simul_params['plot'] and iteration % draw_freq == 0:
            sp_regul = simul_params['regularization']
            sp_step = simul_params['gd_step_size']
            simul_params['regularization'] = solver.alpha
            simul_params['gd-step-size'] = solver.gd_step

            plot_intermediate(L, L_init, iteration, grads, context, simul_params, \
                w2_dists = [w2_iters, w2_dists] if simul_params['track-w2'] else None, \
                    push=(mesh, push), bary_density=bary_density if simul_params['dimension'] == 2 else None)

            simul_params['regularization'] = sp_regul

            if simul_params['track-w2'] and simul_params['dimension'] == 1 and len(w2_dists) > 0:
                axs = context['axs']
                xoff = np.linspace(w2_iters[0], w2_iters[-1], num=n_mesh)
                if not w2_with_samples:
                    axs[2].plot(xoff, current_density(mesh), 'r')
                    axs[2].plot(xoff, bary_density(mesh), 'gray')

            if simul_params['film']:
                context['writer'].grab_frame()
        
        if iteration > 2 and simul_params['check-converged']:
            prev_L = particle_trajectories[-2]
            ratio = np.mean(np.abs((L - prev_L) / prev_L))

            if not simul_params['plot'] and iteration % draw_freq == 0:
                print('\nGradients: ', np.mean(grads), '±', np.std(grads), 'Conv. Ratio: ', ratio/simul_params['convergence-threshold'])
        
            if (not solver.frozen_step or solver.frozen_step and iteration > 200) and ratio < simul_params['convergence-threshold']:
                print('Converged out after {} iterations.'.format(iteration+1))
                print('(Iter.final)  Solver step: {}'.format(solver.gd_step))
                break
    bar.finish()
    
    if simul_params['plot']:
        #if simul_params['film']:
        #    context['writer'].finish()
        plot_free()
        print('I\'m done.')

    
    if w2_marginals is None:
        return L_init, L, np.array(particle_trajectories), (w2_iters, w2_dists), intc_values
    return L_init, L, np.array(particle_trajectories), (w2_iters, w2_dists), intc_values, w2_to_marginals_estimates


# ----
# |
# | SIMULATION BEGINS
# |
# ----
glob_bary_mean, glob_bary_cov = None, None

def run(run_index=0):
    save_bundle = {}

    w2_estim = None
    if simul_params['laws'] in ['norms', 'norm-arctan', 'norm-exp']:
        _, push, _, bary_true_density = laws_2norm_family(means, stds, bary_coords[0], simul_params)
        if simul_params['track-mmd']:
            print('Sampling from barycenter...')
            bary_true_density_ = sample_cdf(lambda x: bary_true_density.integrate_box_1d(-np.inf, x), mesh.min(), mesh.max(), 2*simul_params['samples-per-marginal'], tolerance=1e-3)
            print('Sampling from barycenter... [Ok]')
    elif simul_params['track-w2'] and simul_params['laws'] in ['2d-norm', '2d-n-norms']:
        push = None

        dim = simul_params['dimension']
        bary_mean, bary_cov, _ = utils.barycenter_of_gaussians(means, stds, bary_coords)
        bary_true_density = np.random.multivariate_normal(bary_mean, bary_cov, size=10000)

        if simul_params['track-mmd']:
            bary_true_density_ = np.random.multivariate_normal(bary_mean, bary_cov, size=2*simul_params['samples-per-marginal'])
    elif simul_params['track-w2'] and simul_params['laws'] == 'norm-mixture':
        import mixture_bary_lp as mbp

        bary_pdf, trans_map, bary_kde = mbp.barycenter_density_gaussian_mixture(bary_coords, \
            {'means': means[1], 'sigmas': stds[1], 'weights': means[-1]},
            {'mean': means[0], 'sigma': stds[0]}, mesh, num_samples=10000, verbose=True)
        num_resamples = simul_params['samples-per-marginal']
        if simul_params['track-mmd']:
            num_resamples *= 2
        bary_true_density = bary_kde.resample(size=num_resamples).T
        if simul_params['track-mmd']:
            bary_true_density_ = bary_true_density

        push = None
    elif simul_params['laws'] == 'd-norms':
        push = None

        bary_mean, bary_cov, _ = utils.barycenter_of_gaussians(means, stds, bary_coords)
        bary_true_density = np.random.multivariate_normal(bary_mean, bary_cov, size=2*simul_params['samples-per-marginal'])

        if simul_params['track-mmd']:
            bary_true_density_ = bary_true_density

        global glob_bary_mean
        global glob_bary_cov
        glob_bary_mean = bary_mean
        glob_bary_cov = bary_cov
    else:
        bary_true_density = None
        bary_true_density_ = None
        push = None

    if not simul_params['estimate-w2']:
        init_samples, swarm, trajectories, w2_dists, intc_values = \
            barygd(
                gd_context, 
                simul_params,
                run=run_index, 
                bary_density=bary_true_density_ if simul_params['track-mmd'] else bary_true_density,
                push=push,
                w2_with_samples=True)
    else:
        init_samples, swarm, trajectories, w2_dists, intc_values, w2_diffs = \
            barygd(
                gd_context, 
                simul_params, 
                run=run_index, 
                bary_density=bary_true_density_ if simul_params['track-mmd'] else bary_true_density, 
                push=push, 
                w2_with_samples=not simul_params['track-mmd'], 
                w2_marginals=w2_marginals
                )
        
    print('Experiment {} finished a run.'.format(exp_name))
    print('Run ({}): Saving bundle... [Creating bundle]'.format(run_index))

    save_bundle['init-samples'] = np.copy(init_samples).tolist()
    save_bundle['samples'] = np.copy(swarm).tolist()
    save_bundle['trajectories'] = np.copy(trajectories).tolist()
    save_bundle['w2-iterations-distances'] = w2_dists
    save_bundle['cost-integral'] = intc_values
    if 'norms' in simul_params['laws']:
        if simul_params['dimension'] == 1:
            opt_integral = utils.opt_intc_norms_1d(means, stds, bary_coords)
        else:
            opt_integral = utils.opt_intc_norms(means, stds, bary_coords)
        save_bundle['gaussian-optimal-integral'] = opt_integral
        if simul_params['track-integral']:
            print('Run ({}): Optimal Gaussian barycenter value: {:>10e}'.format(run_index, opt_integral))

    if simul_params['dimension'] == 1:
        init_samples = np.sum(init_samples.T.dot(bary_coords), axis=0).reshape((-1,))
        pushforward = np.sum(swarm.T.dot(np.array(bary_coords)), axis=0)
        samples = np.copy(swarm)
        swarm = pushforward
        valid_samples = swarm[np.where(np.logical_and(swarm < mesh.max(), swarm > mesh.min()))[0]]
        bary_density = S.gaussian_kde(valid_samples.reshape((-1,)))

        save_bundle['init-barycenter-samples'] = np.copy(init_samples).tolist()
        save_bundle['barycenter-samples'] = np.copy(valid_samples).tolist()

        print('n_valid_samples:', valid_samples.size)
    else:
        init_samples = np.sum(np.array([init_samples[k]*bary_coords[k] for k in range(simul_params['n_marginals'])]), axis=0)
        pushforward = np.sum(np.array([swarm[k]*bary_coords[k] for k in range(simul_params['n_marginals'])]), axis=0)

        save_bundle['init-barycenter-samples'] = np.copy(init_samples).tolist()
        save_bundle['barycenter-samples'] = np.copy(pushforward).tolist()
    
    if simul_params['estimate-w2']:
        save_bundle['w2-marginals'] = w2_marginals
        save_bundle['w2-differences'] = w2_diffs

    with open('../img/{}/save_bundle_{}.log'.format(exp_name, run_index), 'w') as f:
        f.write(json.dumps(save_bundle))
        f.write('\r\n')
    print('Run ({}): Saving bundle... [Bundle freed]'.format(run_index))
    del save_bundle
    print('Run ({}): Saving bundle... [Ok]'.format(run_index))

    if simul_params['laws'] in ['2d-norm', '2d-3-norm', '2d-sticks', '2d-n-norms']:
        fig = plt.figure(figsize=(12, 8))
        if simul_params['laws'] == '2d-norm':
            plt.xlim(-3, 3)
            plt.ylim(-3, 3)

        densities = []
        for k in range(simul_params['n_marginals']):
            if simul_params['laws'] in ['2d-norm', '2d-3-norm']:
                densities.append(lambda x, y, k=k: np.exp(-0.5*(istds[k][0,0]*(x-means[k][0])**2 + 2*istds[k][0,1]*(x-means[k][0])*(y-means[k][1]) + istds[k][1,1]*(y-means[k][1])**2)) / (2*np.pi*det(stds[k])))

        plt.title(r'Samples and Density Estimate ($\alpha = {:>3e}$, $h = {:>3e})$'.format(simul_params['regularization'], iss0))
        plt.grid()
        plt.scatter(init_samples[:, 0], init_samples[:, 1], c='lime', marker='o', edgecolor='black', alpha=0.4, label='Initial Samples', zorder=5)
        plt.scatter(pushforward[:, 0], pushforward[:, 1], c='r', marker='o', edgecolor='black', label='Estimated Barycenter Samples', zorder=10)
        sea.kdeplot(x=pushforward[:, 0], y=pushforward[:, 1], fill=True, color='r', alpha=0.8, shadelowest=True, bw_method='silverman', bw_adjust=1.6)

        marginal_samples = []
        X, Y = np.meshgrid(mesh, mesh) if simul_params['laws'] != '2d-sticks' else np.meshgrid(np.linspace(-1, 4, 300), np.linspace(-1, 3, 300))
        label_patches = []
        for k in range(simul_params['n_marginals']):
            #if simul_params['laws'] != '2d-sticks':
            #    samples = np.random.multivariate_normal(means[k], stds[k], size=10000)
            #    marginal_samples.append(samples)
            sea.kdeplot(x=swarm[k, :, 0], y=swarm[k, :, 1], cmap='gray', fill=True, alpha=0.3)
            if simul_params['laws'] != '2d-sticks':
                dens = densities[k](X, Y)
            else:
                dens = fat_stick(X, Y, A[k], B[k], 10, eta=eta)
            plt.contour(X, Y, dens, colors='white', alpha=0.5, extend='neither', levels=3)
        if simul_params['laws'] != '2d-sticks':
            #marginal_samples = np.array(marginal_samples)
            #true_bary_samples = np.sum(np.array([marginal_samples[k]*bary_coords[k] for k in range(simul_params['n_marginals'])]), axis=0)
            dim = simul_params['dimension']
            bary_mean, bary_cov, to = utils.barycenter_of_gaussians(means, stds, bary_coords)
            def bdt(x, y):
                z = np.array([x, y]) - bary_mean
                return np.exp(-0.5*z.T.dot(inv(bary_cov)).dot(z)) / np.sqrt((2*np.pi*det(bary_cov))**dim)
            bdt = np.vectorize(bdt)
            plt.contour(X, Y, bdt(X, Y), colors='black', alpha=0.7, extend='neither', levels=3)
            label_patches.append(mpatches.Patch(color='black', label='Barycenter Density'))
            label_patches.append(mpatches.Patch(color='gray', label='Estimated Marginals (True as white contour lines)'))
        label_patches.append(mpatches.Patch(color='red', label='Barycenter Density Estimate'))

        bt = []
        for t in range(trajectories.shape[0]):
            bt.append(np.sum(np.array([trajectories[t, k]*bary_coords[k] for k in range(simul_params['n_marginals'])]), axis=0))
        bt = np.array(bt)
        for k in range(simul_params['samples-per-marginal']):
            plt.plot(bt[:, k, 0], bt[:, k, 1], '-', c='gray', alpha=0.1, zorder=1)

        ax = plt.gca()
        ax.set_aspect('equal')
        label_handles, _ = ax.get_legend_handles_labels()
        label_handles = label_handles + label_patches
        ax.legend(handles = label_handles)
        print('Saving final samples...')
        fig.savefig('../img/{}/{}_{}.png'.format(exp_name, simul_params['algorithm'], run_index))
        print('Saving final samples... [Ok]')

        if simul_params['plot']:
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
        for k in range(simul_params['samples-per-marginal']):
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
        print('Saving final samples...')
        fig.savefig('../img/{}/{}_{}.png'.format(exp_name, simul_params['algorithm'], run_index))
        print('Saving final samples... [Ok]')

        if simul_params['plot']:
            plt.show()


    if simul_params['laws'] in ['norms', 'norm-arctan', 'norm-exp'] \
        or simul_params['laws'] == 'd-norms' and glob_dim == 1:
        fig, axs = plt.subplots(nrows=2, ncols=1)
        origin, push, pushforward, bary_true_density = laws_2norm_family(means, stds, bary_coords[0], simul_params)

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
        axs[0].hist(valid_samples, density=True, color='r', alpha=0.05, bins='auto')

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
        green, red = np.array([0, 0.7, 0]), np.array([0.4, 0, 0])
        axs[1].scatter(trajectories[0, 0], trajectories[0, 1], marker='o', c='g', edgecolor='gold', alpha=0.3, zorder=5)
        for k in range(simul_params['samples-per-marginal']):
            path = trajectories[:, :, k].reshape(-1, 2)
            for t in range(path.shape[0]-1):
                s = t / path.shape[0]
                axs[1].plot(
                    [path[t, 0], path[t+1, 0]], 
                    [path[t, 1], path[t+1, 1]], '-', 
                    c=utils.hue_lerp(s, green, red),
                    markeredgecolor='gold', 
                    alpha=max(0.2, 1 - np.exp(-0.5*s)), 
                    zorder=1
                    )
        axs[1].scatter(samples[0, valid], samples[1, valid], marker='o', c='r', edgecolor='gold', zorder=10)

        plt.text(x=1.7, y=1.2, s='Endpoints', c='darkred')
        for i in range(100):
            plt.plot([1.8, 1.8], 2*(np.array([i/100, (i+1)/100])-0.5), c=utils.hue_lerp((i+1)/100, green, red), linewidth=20, zorder=60)
        plt.text(x=1.7, y=-1.3, s='Initial points', c='darkgreen')


        if simul_params['laws'] == 'norm-arctan':
            axs[1].set_xlim(-2, 2)
        else:
            axs[1].set_xlim(-6, 6)
        axs[1].set_ylim(-1.5, 1.5 if simul_params['laws'] == 'norm-arctan' else 10)
        axs[1].grid()

        fig.set_size_inches(20, 16)
        print('Saving final samples...')
        fig.savefig('../img/{}/{}.png'.format(exp_name, simul_params['algorithm'], run_index))
        print('Saving final samples... [Ok]')

        if simul_params['plot']:
            plt.show()

    if simul_params['laws'] in ['norm-mixture']:
        fig, axs = plt.subplots(nrows=2, ncols=1)

        origin = lambda x: np.exp(-0.5*(x - means[0])**2 / stds[0]**2) / (stds[0] * np.sqrt(2*np.pi))
        pushforward = np.vectorize(utils.gaussian_mixture(means[-1], means[1], stds[1]))

        axs[0].set_title('Samples and Density Estimate (Marginals in grey)')
        axs[0].set_xlim(np.min(np.min(trajectories)), np.max(np.max(trajectories)))
        axs[0].scatter(init_samples, np.zeros(init_samples.size), c='b', label='Initial Samples')
        axs[0].scatter(valid_samples, np.zeros(valid_samples.size), c='r', label='BARYGD Samples')
        axs[0].plot(mesh, bary_density.pdf(mesh), c='b', alpha=0.7, label='Estimated Density')
        axs[0].plot(mesh, origin(mesh), c='gray', alpha=0.3)
        axs[0].plot(mesh, pushforward(mesh), c='gray', alpha=0.3)
        axs[0].hist(valid_samples, density=True, color='r', alpha=0.05, bins='auto')
        axs[0].plot(mesh, bary_pdf(mesh), c='black', linewidth=2, alpha=0.9, label='Barycenter Density')

        exp_label = r'$\mathrm{bar}(\mathscr{N}(\mu_1,1), \mu_{mixture})$'
        axs[0].grid()
        axs[0].legend()

        axs[1].set_title(r'Evolution of $X_t$ (Initial samples in green)')
        #axs[1].set_aspect('equal')

        valid_1 = np.where(np.logical_and(samples[0] < mesh.max(), samples[0] > mesh.min()))[0]
        valid_2 = np.where(np.logical_and(samples[1] < mesh.max(), samples[1] > mesh.min()))[0]
        valid = np.intersect1d(valid_1, valid_2)

        # ASSUMPTION: no point exploded
        green, red = np.array([0, 0.7, 0]), np.array([0.4, 0, 0])
        axs[1].scatter(trajectories[0, 1], trajectories[0, 0], marker='o', c='g', edgecolor='gold', alpha=0.3, zorder=5)
        for k in range(simul_params['samples-per-marginal']):
            path = trajectories[:, :, k].reshape(-1, 2)
            for t in range(path.shape[0]-1):
                s = t / path.shape[0]
                axs[1].plot(
                    [path[t, 1], path[t+1, 1]], 
                    [path[t, 0], path[t+1, 0]], '-', 
                    c='gray',
                    markeredgecolor='gold', 
                    alpha=max(0.2, 1 - np.exp(-3.5*s)), 
                    zorder=1
                    )
        axs[1].scatter(samples[1, valid], samples[0, valid], marker='o', c='r', edgecolor='gold', zorder=10)
        axs[1].plot(mesh, [trans_map(x) for x in mesh], c='gold', linewidth=2)

        #plt.text(x=5.25, y=4.5, s='Endpoints', c='darkred')
        #for i in range(100):
        #    plt.plot([5.5, 5.5], 8*(np.array([i/100, (i+1)/100])-0.5), c=utils.hue_lerp((i+1)/100, green, red))
        #plt.text(x=5.15, y=-4.8, s='Initial points', c='darkgreen')

        axs[1].set_xlim(-5, 6)
        axs[1].set_ylim(-6, 6)
        axs[1].grid()

        fig.set_size_inches(20, 16)
        print('Saving final samples...')
        fig.savefig('../img/{}/{}.png'.format(exp_name, simul_params['algorithm'], run_index))
        print('Saving final samples... [Ok]')

        if simul_params['plot']:
            plt.show()
            
    if simul_params['track-integral']:
        plt.clf()
        plt.title(r'Evolution of $\int c d\gamma_t$ over time $t$')
        plt.xlabel(r'Iterations $t$')
        plt.ylabel(r'Value of $\int c d\gamma_t$')
        plt.grid()
        plt.plot(np.arange(len(intc_values))*draw_freq, intc_values, 'gray')
        print('Saving integral curve plot...')
        plt.savefig('../img/{}/intc_{}.png'.format(exp_name, run_index))
        print('Saving integral curve plot... [Ok]')
        if simul_params['plot']:
            plt.show()

    if not simul_params['estimate-w2']:
        return w2_dists, intc_values
    return w2_dists, intc_values, w2_diffs


if __name__ == '__main__':
    try:
        if simul_params['mode'] == 'run':
            run()
        else:
            w2_estimates = []

            if simul_params['mode'] == 'dependance-regularization':
                n_runs = 10
                param_values = [10**i for i in range(n_runs)]
                if simul_params['kernel-order'] == 2:
                    iter_values = [3000]*n_runs
                    step_values = [1/(1e3*p) for p in param_values]
                else:
                    iter_values = [1000]*n_runs
                disp_name = 'regul.'
                plot_name = r'$1/\sqrt{\alpha}$'
            elif simul_params['mode'] == 'dependance-samples':
                param_values = [50, 100, 200]
                n_runs = len(param_values)
                disp_name = 'samples'
                plot_name = 'number of samples'
            elif simul_params['mode'] == 'dependance-iterations':
                n_runs = 1
                param_values = [0]
                disp_name = 'W2-Iter'
                plot_name = 'number of iterations'
            elif simul_params['mode'] == 'test':
                n_runs = 1
                param_values = [0]
                simul_params['n_iterations'] = 1
                disp_name = 'test1'
                plot_name = 'test2'

            for i in range(n_runs):
                if simul_params['mode'] == 'dependance-regularization':
                    simul_params['regularization'] = param_values[i]
                    simul_params['n_iterations'] = iter_values[i]
                    simul_params['initial-gd-step-size'] = step_values[i]
                    simul_params['gd_step_size'] = step_values[i]
                elif simul_params['mode'] == 'dependance-samples':
                    simul_params['samples-per-marginal'] = param_values[i]
                
                if not simul_params['estimate-w2']:
                    w2_dists, intc_values = run(run_index=i)
                else:
                    w2_dists, intc_values, _ = run(run_index=i)

                if simul_params['track-w2'] and len(w2_dists) > 0:
                    w2 = w2_dists[1][-1]
                    w2_estimates.append(w2)
                    print('(Run {}/{})  Wasserstein distance ({}: {:e}): {:e}'.format(i+1, n_runs, disp_name, param_values[i], w2))

            
            if simul_params['track-w2']:
                print('Saving W2 estimates...')
                np.save('../img/{}/DEP_{}.npy'.format(exp_name, simul_params['film_name']), np.array(w2_estimates))
                if simul_params['mode'] == 'dependance-iterations':
                    np.save('../img/{}/DEP_{}-W2dists.npy'.format(exp_name, simul_params['film_name']), np.array(w2_dists))
                print('Saving W2 estimates... [Ok]')

                print('Plotting experiment result...')
                plt.clf()
                fig = plt.figure(1)

                plt.title('Wasserstein distance as a function of {}'.format(plot_name))
                plt.xlabel(plot_name)
                plt.ylabel(r'$W_2(T {}_{\#} \gamma_{' + simul_params['algorithm'] + r'},\, T {}_{\#} \gamma^\star)$')

                plt.grid()
                if simul_params['mode'] == 'dependance-regularization':
                    plt.plot(1 / np.sqrt(param_values), w2_estimates)
                elif simul_params['mode'] == 'dependance-iterations':
                    iter_indices = [j for j in range(simul_params['n_iterations']) if j > 0 and j % w2_freq == 0]
                    plt.plot(iter_indices, w2_dists[-1], c='gray')
                else:
                    plt.plot(param_values, w2_estimates)

                print('Saving experiment plot...')
                fig.savefig('../img/{}/DEP_{}.png'.format(exp_name, simul_params['film_name']))
                print('Saving experiment plot... [Ok]')

                print('Plotting experiment result... [Stop]')
                plt.show()
                print('Experiment « {} » ended.'.format(exp_name))
    except:
        print('Failed with error:')
        raise
