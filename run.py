import pdb
import os
import sys
import argparse
import logging
from execution import Execution
from distance import Distance
from descent import Descent, SolverExplosion
from plotter import Plotter
import numpy as np
from progress.bar import Bar
from utils import make_dir
import datetime


class Run:
    def __init__(self, args):
        self.name = self.name_from_args(args)

        self.log = not args.no_logging
        self.initialize_logger(args)

        self.assert_args_correct(args)
        self.execution = Execution.from_args(args)
        self.execution.name = self.name

        self.plotter = Plotter(self.execution, not args.dont_show)


    def initialize_logger(self, args):
        if self.log:
            folder = os.path.join('runs', self.name)
            make_dir(folder)
            logfile = os.path.join(folder, 'log')
            handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(filename=logfile)]
        else:
            handlers = [logging.StreamHandler(sys.stdout)]

        logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            handlers=handlers,
            encoding='utf-8', 
            level=logging.DEBUG
        )
        logging.info('The logger is set up.\n\tRun name: {} .'.format(self.name))
        logging.info('[Argparse environment]:\n\t' + str(args))


    def check_name(self):
        if self.name != self.execution.name:
            self.execution.name = self.name

            logging.info('[RENAMING] Changed execution name \'{}\' .'.format(self.execution.name))


    def initialize_problem(self, ref_dist='wvar', only_target=False):
        self.descent = Descent(self.execution, ref_dist)
        self.descent.make_problem(only_target=only_target)

        self.distance = Distance(self.execution, self.execution.distance_modes)
        self.execution.iteration_records = []


    # TODO: last_value = f(d) pour d = 10, 100, 1000
    def loop(self):
        assert self.descent.problem_initialized, 'Cannot start optimization if the problem is not set up.'
        start_iteration = self.execution.start_iteration

        self.progress = Bar(
            'Optim.:', 
            max=self.execution.num_iterations_per_solver, 
            suffix='ETA %(eta)d sec, %(index)d/%(max)d - %(percent)d %%'
        )
        self.progress.index = start_iteration

        for iteration in range(start_iteration, self.execution.num_iterations_per_solver):
            self.descent.step(iteration)

            if (iteration % self.execution.save_samples_frequency == 0) or (iteration == self.execution.num_iterations_per_solver-1):
                barycenter_samples = [
                    Run.samples_from_coupling_(
                        s.coupling, 
                        self.execution.weights, 
                        self.execution.num_marginals
                    ) 
                    for s in self.execution.solvers
                ]
                self.execution.barycenter_samples_records.append(barycenter_samples)
                self.execution.iteration_records.append(iteration)
                self.descent.update_alpha(iteration)
                self.distance.step()
                self.execution.iteration = iteration

                if iteration == self.execution.num_iterations_per_solver-1:
                    for s in range(self.execution.num_solvers):
                        self.execution.solvers[s].steps_history.append(self.execution.solvers[s].gd_step)

            self.execution.start_iteration = iteration
            self.progress.next()

        self.progress.finish()
        self.save()


    def samples_from_coupling_(coupling, weights, num_marginals):
        samples = np.sum(
            np.array([weights[i]*coupling[i, :] for i in range(num_marginals)]), 
            axis=0
        )
        return samples

    
    def save(self):
        self.execution.save()
        logging.info('[RUN SAVE] Saved the execution.')

    
    def compute_distances(self):
        self.distance.compute_average_curves()


    def plot_and_save_distances(self):
        self.plotter.plot_and_save_distances(self.name, self.execution.distance_modes)


    def plot_and_save_steps(self):
        self.plotter.plot_and_save_steps(self.name)


    def plot_and_save_2D_state(self):
        self.plotter.plot_and_save_2D(self.name)

    
    def plot_and_save_2D_movie_grads(self):
        #assert self.execution.dimension == 2, 'Cannot plot particles movie in dim != 2.'
        self.plotter.plot_and_save_2D_movie_grads(self.name)


    def assert_args_correct(self, args):
        if args.start_from_diagonal or args.start_from_marginals:
            xor = lambda a, b: (a and (not b)) or ((not a) and b)
            assert xor(args.start_from_diagonal, args.start_from_marginals), 'Options --start-from-diagonal and --start-from-marginals are mutually exclusive.'
        if args.plot_final:
            assert args.dimension == 2, 'Option --plot-final is only available when dimension=2.'


    def name_from_args(self, args):
        technical_name = args.technical_name

        if technical_name == '':
            now = datetime.datetime.now()
            make_two_digits = lambda x: x if len(x) == 2 else '0'+x

            technical_name = '{}{}{}{}{}-({}-{}-{}-{})nNds-({})w-({}-{})isia-({}-{}-{})itsfaf-{}rs'.format(
                make_two_digits(str(now.day)),
                make_two_digits(str(now.month)),
                str(now.year)[-2:],
                make_two_digits(str(now.hour)),
                make_two_digits(str(now.minute)),
                args.num_marginals,
                args.num_particles_per_marginal,
                args.dimension,
                args.num_solvers,
                args.weights,
                args.initial_step,
                args.initial_alpha,
                args.num_iter,
                args.save_samples_frequency,
                args.alpha_update_frequency,
                args.random_seed
            )
        if args.name == '':
            name = ('MMD-' if args.solver_kernel == 'mmd' else 'SVGD-') + technical_name
        else:
            name = ('MMD-' if args.solver_kernel == 'mmd' else 'SVGD-') + technical_name + '-' + args.name

        return name
