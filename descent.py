import pdb
import os
import sys
import copy
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmd_kernel import MMDKernel
from kernel import SVGDKernel
from solver import Solver
from utils import barycenter_of_gaussians, sample_uniform_dsphere
from scipy.linalg import sqrtm
from numpy.linalg import inv


class SolverExplosion(Exception):
    pass


class Descent:
    def __init__(self, execution, ref_dist):
        self.ref_dist = ref_dist
        self.execution = execution
        self.problem_initialized = False
        self.restart_counts = [0]*self.execution.num_solvers


    def step(self, iteration):
        solvers = self.execution.solvers

        for s in range(self.execution.num_solvers):
            solvers[s].gd_step = self.step_schedule(self.execution.solver_decay[s], s)
            solvers[s].steps_history.append(solvers[s].gd_step)

            if solvers[s].exploded:
                logging.critical('\nSolver {} exploded (out of box of width {}).'.format(s, solvers[s].explosion_value))
                raise SolverExplosion()

            solvers[s].update()
            self.execution.solver_decay[s] = self.execution.solver_decay[s] + 1
            solvers[s].alphas.append(solvers[s].alpha)


    def update_alpha(self, iteration):
        solvers = self.execution.solvers

        for s in range(self.execution.num_solvers):
            if solvers[s].dynindex < solvers[s].dynthreshold:
                self.restart_counts[s] += 1
                solvers[s].alpha = self.alpha_schedule(self.restart_counts[s])

                logging.info(
                    '[ALPHA STEP] Solver {}: (alpha, step, iter.): ({}, {}, {})'.format(
                        s, solvers[s].alpha, solvers[s].gd_step, iteration
                    )
                )
                symbol = '>' if solvers[s].dynindex > solvers[s].dynthreshold else '<'
                logging.info(
                    'Dynamics measure (Solver {}): {} {} {}'.format(
                        s, solvers[s].dynindex, symbol, solvers[s].dynthreshold
                    )
                )


    def alpha_schedule(self, t):
        return self.execution.initial_alpha + t


    def step_schedule(self, t, s):
        grads = self.execution.solvers[s].grads
        solver = self.execution.solvers[s]

        step = solver.gd_step
        if t < 3: return step

        grad_n = grads[1]
        grad_nn = grads[0]
        grad_nn_norm = np.sqrt(np.sum(grads[0]**2, axis=(0, 2))).mean()

        deltaf = np.sqrt(np.sum((grad_n - grad_nn)**2, axis=(0, 2))).mean()/grad_nn_norm
        #step = np.min([step, self.execution.deltacoeff/deltaf, 1/solver.alpha])
        step = np.min([step, 1/solver.alpha])

        solver.deltafs.append(1/deltaf)
        solver.gradnorm.append(grad_nn_norm)
        return step


    def make_target_barycenter(self):
        def random_covariance(scale, dim):
            a = scale*(np.random.rand(dim, dim) - 0.5) / np.sqrt(dim)
            pd = 0.1*np.eye(dim) + a.dot(a.T)
            return pd
        random_diagonal_covariance = lambda scale, dim: np.diag(scale*(np.random.rand(dim) + 1))
        rand_pd_matrix = random_covariance

        if rand_pd_matrix == random_covariance:
            logging.info('Using random_covariance to generate the covariance matrices.')
        elif rand_pd_matrix == random_diagonal_covariance:
            logging.info('Using random_diagonal_covariance to generate the covariance matrices.')

        def gaussian_potential_gradient(mean, sigma):
            inv_sigma = np.linalg.inv(sigma)
            pot = lambda x, mean=mean, inv_sigma=inv_sigma: inv_sigma.dot(x - mean)
            return pot

        directions = sample_uniform_dsphere(self.execution.dimension, self.execution.num_marginals)
        scales = (np.random.rand(self.execution.num_marginals) + 0.5)*self.execution.mean_scale
        self.execution.means = np.array([directions[i]*scales[i] for i in range(self.execution.num_marginals)])

        logging.info('Sampled the means of the marginals.')
        self.execution.covariances = np.array([rand_pd_matrix(self.execution.covariance_scale, self.execution.dimension) for _ in range(self.execution.num_marginals)])

        #
        self.execution.cov_eigvals_ratios = np.array([np.min(self.execution.covariances[i])/np.max(self.execution.covariances[i]) for i in range(self.execution.num_marginals)])
        self.execution.mean_cov_eigvals_ratio = np.sum([self.execution.weights[i]*np.min(self.execution.covariances[i])/np.max(self.execution.covariances[i]) for i in range(self.execution.num_marginals)])
        #

        logging.info('Sampled the covariances of the marginals.')
        self.execution.potential_gradients = [gaussian_potential_gradient(self.execution.means[i], self.execution.covariances[i]) for i in range(self.execution.num_marginals)]
        logging.info('Created potential gradients for the marginals.')

        gaussian_barycenter_mean, gaussian_barycenter_covariance, _ = barycenter_of_gaussians(self.execution.means, self.execution.covariances, self.execution.weights)
        self.execution.gaussian_barycenter_mean = gaussian_barycenter_mean
        self.execution.gaussian_barycenter_covariance = gaussian_barycenter_covariance
        logging.info('Computed exact barycenter mean and covariance.')


    def make_nongaussian_barycenter(self):
        def random_covariance(scale, dim):
            a = scale*(np.random.rand(dim, dim) - 0.5) / np.sqrt(dim)
            pd = 0.1*np.eye(dim) + a.dot(a.T)
            return pd
        random_diagonal_covariance = lambda scale, dim: np.diag(scale*(np.random.rand(dim) + 1))
        rand_pd_matrix = random_covariance

        if rand_pd_matrix == random_covariance:
            logging.info('Using random_covariance to generate the covariance matrices.')
        elif rand_pd_matrix == random_diagonal_covariance:
            logging.info('Using random_diagonal_covariance to generate the covariance matrices.')

        self.execution.means = np.zeros((self.execution.num_marginals, self.execution.dimension))
        self.execution.gaussian_barycenter_mean = np.zeros((self.execution.dimension))
        self.execution.gaussian_barycenter_covariance = np.eye(self.execution.dimension) ##
        barycenter_potential_gradient = lambda x: x ##

        bary_cov = self.execution.gaussian_barycenter_covariance
        bary_cov_sqrt = sqrtm(bary_cov)
        bary_cov_isqrt = inv(bary_cov_sqrt)
        ti_cov = np.array([rand_pd_matrix(self.execution.covariance_scale, self.execution.dimension) for _ in range(self.execution.num_marginals-1)])
        ti = []
        for i in range(self.execution.num_marginals-1):
            ti.append(bary_cov_isqrt.dot(bary_cov_sqrt.dot(ti_cov[i]).dot(bary_cov_sqrt)).dot(bary_cov_isqrt))
        sum_ti = np.zeros((self.execution.dimension, self.execution.dimension))
        for t in ti:
            sum_ti = sum_ti + self.execution.weights[i]*t
        ti.append((np.eye(self.execution.dimension) - sum_ti) / self.execution.weights[-1])
        ti_inv = [inv(t) for t in ti]
        self.execution.covariances = np.array([t.dot(bary_cov).dot(bary_cov) for t in ti])

        def potential_gradient(tinv):
            grad = lambda x: tinv.dot(barycenter_potential_gradient(tinv.dot(x)))
            return grad
        self.execution.potential_gradients = [potential_gradient(tinv) for tinv in ti_inv]

        logging.info('Created potential gradients for the marginals.')
        logging.info('Stored exact barycenter mean and covariance.')


    def make_problem(self, only_target=False):
        if self.execution.nongaussian:
            self.make_nongaussian_barycenter()
        else:
            self.make_target_barycenter()
        if only_target: return None

        if self.execution.solver_kernel == 'svgd':
            self.kernel = SVGDKernel(bandwidth=self.execution.kernel_scale)
            for gradient in self.execution.potential_gradients:
                self.kernel.append_potential_gradient(gradient)
            self.kernel.subsample_fraction = self.execution.subsample_fraction
            logging.info('The SVGD potential gradients are initialized.')
        else:
            self.marginal_samples = np.array([np.random.multivariate_normal(means[i], covariances[i], size=num_particles_per_marginal) for i in range(num_marginals)])
            self.kernel = MMDKernel(marginal_samples, bandwidth=kernel_scale)
            logging.info('The marginal samples for MMD kernel are sampled.')
        logging.info('The solver ({}) kernel is set up.'.format(self.execution.solver_kernel))

        self.execution.solvers = []

        # TODO: tracer last_value = f(N) et upper_sigma = f(N)

        for s in range(self.execution.num_solvers):
            model_solver = Solver(
                num_marginals=self.execution.num_marginals,
                num_particles_per_marginal=self.execution.num_particles_per_marginal,
                dim=self.execution.dimension,
                gd_step=self.execution.initial_gd_step,
                lambdas=self.execution.weights,
                kernel=self.kernel,
                alpha=self.execution.initial_alpha,
                alpha_freq=self.execution.solver_alpha_freq,
                variable_alpha=False,
                adaptive_step_size=False,
                frozen_step=True,
                threading=False,
                explosion_scale=self.execution.explosion_scale,
                initial_gd_step=self.execution.initial_gd_step,
                dynthreshold=self.execution.dynamics_alpha_threshold,
                #max_iter_subsample=self.execution.num_iterations_per_solver+1 ## hardcoded!!
                max_iter_subsample=self.execution.max_iter_subsample
            )
            model_solver.execution = self.execution

            if self.execution.start_from_marginals or self.execution.start_from_diagonal:
                initial_coupling = np.zeros((self.execution.num_marginals, self.execution.num_particles_per_marginal, self.execution.dimension))
                initialization_string = 'from marginals'

                if self.execution.start_from_diagonal:
                    initialization_string = 'from diagonal'
                    sample = np.random.multivariate_normal(self.execution.means[0], self.execution.covariances[0], size=self.execution.num_particles_per_marginal)

                for marginal in range(self.execution.num_marginals):
                    if self.execution.start_from_marginals:
                        marginal_samples = np.random.multivariate_normal(self.execution.means[marginal], self.execution.covariances[marginal], size=self.execution.num_particles_per_marginal)
                        initial_coupling[marginal] = marginal_samples
                    else:
                        initial_coupling[marginal] = sample.copy()
                
                model_solver.initialize_coupling('from-coupling', initial_coupling)
            else:
                scale = 10/np.sqrt(self.execution.num_particles_per_marginal)
                model_solver.initialize_coupling('uniform', (0, scale))
                #model_solver.initialize_coupling('form-a', [
                #    [-5, -5, -5, -5, -5, -5, -5, -5], 
                #    [0, 5, 10, -10, 2, -2, 3, -3], 
                #])
                initialization_string = 'uniformly in [0, 1]^{}'.format(self.execution.dimension)
            
            self.execution.solvers.append(model_solver)

            logging.info('The coupling is initialized ({}).'.format(initialization_string))
            logging.info('Solver {} (of {}) is set up.'.format(s, self.execution.num_solvers))

        self.execution.barycenter_samples = [None for _ in range(self.execution.num_solvers)]
        self.execution.iteration_records = [] # for these two lists...
        self.execution.barycenter_samples_records = [] # ... as many as [num_solvers*num_iterations_per_solver / save_samples_frequency]

        self.execution.solver_decay = [1] * self.execution.num_solvers
        self.dynamics_restarts = 0

        logging.info('The problem context is initialized.')
        logging.info('Problem: {} solvers with {}*{} particles in dimension {}.'.format(
            self.execution.num_solvers, 
            self.execution.num_marginals, 
            self.execution.num_particles_per_marginal, 
            self.execution.dimension
            ))
            
        self.problem_initialized = True