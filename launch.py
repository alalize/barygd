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


if __name__ == '__main__':
    mmd_kernel_choices = ['gaussian', 'id', 'var', 'idvar']
    MMD_KERNEL_GAUSSIAN = 0
    MMD_KERNEL_MEANS = 1
    MMD_KERNEL_COVARIANCES = 2

    distance_mode_choices = ['id', 'id2', 'var', 'idvar', 'w2ansatz', 'wvar', 'wovar', 'norm2', 'ratiovarw2']

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', default='', type=str, help='Pickle to resume distance computations from.')
    parser.add_argument('--plot-final', action='store_true', help='Plots the marginals and barycenters (true and estimated) in d=2. Requires --resume-from.')
    parser.add_argument('--plot-distance', action='store_true', help='Plots the average distance curve(s) to true barycenter clouds. Requires --resume-from.')
    parser.add_argument('--distances', type=str, nargs='+', choices=distance_mode_choices, default=['idvar', 'wovar'], help='Method for distance computations. Choices: {}.'.format(distance_mode_choices))
    parser.add_argument('--mmd-kernel', type=str, default=mmd_kernel_choices[MMD_KERNEL_MEANS], choices=mmd_kernel_choices, help='Kernel to compute MMD distance estimate. Default: Distance of means. Choices: {}.'.format(mmd_kernel_choices))
    parser.add_argument('--dimension', type=int, default=2)
    parser.add_argument('--num-marginals', type=int, default=2)
    parser.add_argument('--num-particles-per-marginal', type=int, default=50)
    parser.add_argument('--num-solvers', type=int, default=5)
    parser.add_argument('--weights', type=int, nargs='+', default=[0.5, 0.5])
    parser.add_argument('--save-samples-frequency', type=int, default=20)
    parser.add_argument('--num-iter', type=int, default=30*10)
    parser.add_argument('--random-gaussian-barycenter-mean-scale', type=float, default=12)
    parser.add_argument('--random-gaussian-barycenter-covariance-scale', type=float, default=5)
    parser.add_argument('--initial-step', type=float, default=0.1)
    parser.add_argument('--initial-alpha', type=float, default=1)
    parser.add_argument('--kernel-scale', type=float, default=1, help='The kernel bandwidth for the SVGD/MMD kernel.')
    parser.add_argument('--alpha-update-frequency', type=int, default=20)
    parser.add_argument('--random-seed', type=int, default=0)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--technical-name', type=str, default='')
    parser.add_argument('--dont-show', action='store_true', help='If set, will save image but not show the plot.')
    parser.add_argument('--start-from-marginals', action='store_true', help='Starts from a coupling with correct marginals.')
    parser.add_argument('--start-from-diagonal', action='store_true', help='If set, samples one marginal and copies it along the coupling.')
    parser.add_argument('--solver-kernel', type=str, choices=['svgd', 'mmd'], default='svgd', help='The penalization kernel used by the algorithm for the optimization.')
    #parser.add_argument('--dynscal', action='store_true', help='If set, alpha is updated only when the dynamics halts.')
    parser.add_argument('--dynscalth', type=float, default=0.5, help='The threshold below which the dynamics is deemed halted.')
    parser.add_argument('--explosion-scale', type=float, default=50, help='If any coordinate of coupling samples overflows this, an explosion occurred.')
    parser.add_argument('--recover', action='store_true', help='If set, reruns the optimization loop.')
    parser.add_argument('--reference-distance-runs', type=int, default=256, help='Number of samplings to compute average reference distances.')
    parser.add_argument('--take-name', default=False, action='store_true', help='If set, will check_name .')
    parser.add_argument('--no-logging', action='store_true')
    parser.add_argument('--track-dist', type=str, default='wvar', choices=distance_mode_choices, help='The distance to focus on to change alpha and step size.')
    parser.add_argument('--plot-movie', action='store_true', help='If set, generates the movie with the marginal+barycenter particles and the gradients over iterations.')
    parser.add_argument('--deltacoeff', type=float, default=0.5, help='Coefficient in front of gradient Lipschitz constant estimate.')
    parser.add_argument('--subsampling', type=float, default=0.25, help='Proportion of samples to keep for kernel gradient computation when subsampling. At least one sample is guaranteed to be used. To get rid of subsampling, set this to 1.')
    parser.add_argument('--max-subsampling-iter', type=int, default=-1, help='Iteration after which subsampling in kernel gradient computation is disabled.')
    parser.add_argument('--nosubsampling', action='store_true', help='If set, no subsampling is used in the kernel gradient computation.')
    parser.add_argument('--non-gaussian-problem', action='store_true', help='If set, use a non-gaussian barycenter and marginals.')
    args = parser.parse_args()
    np.random.seed(args.random_seed)

    if args.max_subsampling_iter == -1:
        args.max_subsampling_iter = args.num_iter+1
    if args.nosubsampling:
        args.max_subsampling_iter = None

    def fail(run):
        run.save()
        logging.critical(traceback.format_exc())
        quit()

    # TODO: courbe d=10, last_val = f(N) avec n_solvers=10 à 100
    # TODO: courbe last_value(id2, wovar, var) = f(N) avec num_solvers > 1
    try:
        assert 0 <= args.subsampling <= 1, 'Subsampling fraction is not within interval ]0,1].'

        run = Run(args)
        run.initialize_problem(ref_dist=args.track_dist)
        if args.resume_from != '':
            run.execution.load(args.resume_from)
        run.check_name()

        if args.resume_from == '' or (args.resume_from != '' and args.recover):
            run.execution.num_iterations_per_solver = args.num_iter
            run.loop()
        
        if args.plot_distance and args.resume_from == '':
            run.compute_distances()

        if args.plot_distance:
            run.plot_and_save_distances()
            run.plot_and_save_steps()

        if args.plot_final:
            run.plot_and_save_2D_state()
        
        if args.plot_movie:
            run.plot_and_save_2D_movie_grads()
    except Exception:
        fail(run)
    except KeyboardInterrupt:
        fail(run)
    finally:
        logging.critical(traceback.format_exc())
        quit()
