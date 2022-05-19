import pdb
import traceback
import argparse
import logging
import numpy as np
from execution import Execution
from distance import Distance
from descent import Descent, SolverExplosion
from plotter import Plotter
from run import Run
from progress.bar import Bar
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mmd_kernel_choices = ['gaussian', 'id', 'var', 'idvar']
    MMD_KERNEL_GAUSSIAN = 0
    MMD_KERNEL_MEANS = 1
    MMD_KERNEL_COVARIANCES = 2

    distance_mode_choices = ['id', 'var', 'idvar', 'w2ansatz', 'id2', 'wvar', 'sigma']

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', default='', type=str, help='Pickle to resume distance computations from.')
    parser.add_argument('--plot-final', action='store_true', help='Plots the marginals and barycenters (true and estimated) in d=2. Requires --resume-from.')
    parser.add_argument('--plot-distance', action='store_true', help='Plots the average distance curve(s) to true barycenter clouds. Requires --resume-from.')
    parser.add_argument('--distances', type=str, nargs='+', choices=distance_mode_choices, default=['id', 'var'], help='Method for distance computations. Choices: {}.'.format(distance_mode_choices))
    parser.add_argument('--mmd-kernel', type=str, default=mmd_kernel_choices[MMD_KERNEL_MEANS], choices=mmd_kernel_choices, help='Kernel to compute MMD distance estimate. Default: Distance of means. Choices: {}.'.format(mmd_kernel_choices))
    parser.add_argument('--dimension', type=int, default=2)
    parser.add_argument('--num-marginals', type=int, default=2)
    parser.add_argument('--num-particles-per-marginal', type=int, default=50)
    parser.add_argument('--num-solvers', type=int, default=5)
    parser.add_argument('--weights', type=int, nargs='+', default=[0.5, 0.5])
    parser.add_argument('--save-samples-frequency', type=int, default=25)
    parser.add_argument('--num-iter', type=int, default=30*10)
    parser.add_argument('--random-gaussian-barycenter-mean-scale', type=float, default=3)
    parser.add_argument('--random-gaussian-barycenter-covariance-scale', type=float, default=5)
    parser.add_argument('--initial-step', type=float, default=0.1)
    parser.add_argument('--initial-alpha', type=float, default=10)
    parser.add_argument('--damping', type=float, default=1)
    parser.add_argument('--kernel-scale', type=float, default=1, help='The kernel bandwidth for the SVGD/MMD kernel.')
    parser.add_argument('--alpha-update-frequency', type=int, default=20)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--technical-name', type=str, default='')
    parser.add_argument('--dont-show', action='store_true', help='If set, will save image but not show the plot.')
    parser.add_argument('--start-from-marginals', action='store_true', help='Starts from a coupling with correct marginals.')
    parser.add_argument('--solver-kernel', type=str, choices=['svgd', 'mmd'], default='svgd', help='The penalization kernel used by the algorithm for the optimization.')
    parser.add_argument('--dynscal', action='store_true', help='If set, alpha is updated only when the dynamics halts.')
    parser.add_argument('--dynscalth', type=float, default=0.001, help='The threshold below which the dynamics is deemed halted.')
    parser.add_argument('--explosion-scale', type=float, default=20, help='If any coordinate of coupling samples overflows this, an explosion occurred.')
    parser.add_argument('--start-from-diagonal', action='store_true', help='If set, samples one marginal and copies it along the coupling.')
    parser.add_argument('--recover', action='store_true', help='If set, reruns the optimization loop.')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--reference-distance-runs', type=int, default=100, help='Number of samplings to compute average reference distances.')
    args = parser.parse_args()
    args.no_logging = True
    np.random.seed(args.random_seed)

    dimensions = np.array([5*d for d in range(1, 12+1)])
    num_modes, num_dims = len(args.distances), len(dimensions)

    ref_dists, std_dists = {}, {}
    for mode in args.distances:
        ref_dists[mode] = []
        std_dists[mode] = []

    progress = Bar(
        'Calc.:', 
        max=num_dims, 
        suffix='ETA %(eta)d sec, %(index)d/%(max)d - %(percent)d %%'
    )
    for di, dim in enumerate(dimensions):
        args.dimension = dim

        run = Run(args)
        run.initialize_problem(only_target=True)
        run.compute_distances()

        for mode in args.distances:
            ref_dists[mode].append(run.execution.reference_distances[mode])
            std_dists[mode].append(run.execution.std_reference_distances[mode])

        progress.next()
    progress.finish()

    for mode in args.distances:
        ref_dists[mode] = np.array(ref_dists[mode])
        std_dists[mode] = np.array(std_dists[mode])

    colors = ['red', 'blue', 'green', 'black', 'gold']
    plt.title(r'$N=' + str(args.num_particles_per_marginal) + r'$ particles')
    plt.xlabel('dimension')
    plt.ylabel('ref. squared distance')
    plt.grid()
    for mi, mode in enumerate(args.distances):
        plt.plot(dimensions, ref_dists[mode], c=colors[mi], label=mode)

        upper = ref_dists[mode] + 2*std_dists[mode]
        lower = ref_dists[mode] - 2*std_dists[mode]
        plt.fill_between(dimensions, lower, upper, color=colors[mi], alpha=0.1)
        plt.plot(dimensions, lower, c=colors[mi], alpha=0.2)
        plt.plot(dimensions, upper, c=colors[mi], alpha=0.2)
    plt.legend()

    fig = plt.gcf()
    inOfcm = lambda cm: cm/2.54
    fig.set_size_inches(inOfcm(24), inOfcm(18))

    if args.save:
        plt.savefig('refs-depOnDim-{}N.png'.format(args.num_particles_per_marginal))
    plt.show()
