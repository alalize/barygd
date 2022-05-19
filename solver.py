from os import stat
import pdb
import numpy as np
from numpy.lib.function_base import gradient
from numpy.lib.type_check import _nan_to_num_dispatcher
import utils
import scipy.stats as stats


class Solver:
    def __init__(self, num_marginals, num_particles_per_marginal, dim, gd_step,\
        lambdas, kernel, alpha, adaptive_step_size=False, threading=False, variable_alpha=False, alpha_freq=500,\
            frozen_step=False, coulomb=False, use_c=False, skip_marginals=[], grad_func=None, explosion_scale=20,
            initial_gd_step=0.1, dynthreshold=0, max_iter_subsample=None):
        self.num_marginals = num_marginals
        self.num_particles_per_marginal = num_particles_per_marginal
        self.dim = dim
        self.gd_step = gd_step
        self.initial_gd_step = initial_gd_step
        self.lambdas = lambdas
        self.kernel = kernel
        self.alpha = alpha
        self.variable_alpha = variable_alpha
        self.coupling = None
        self.adaptive_step_size = adaptive_step_size
        self.threading = threading
        self.initialized = False
        self.iteration = 0
        self.dynindex_coupling = 0
        self.alpha_freq = alpha_freq
        self.frozen_step = frozen_step
        self.use_c = use_c
        self.dc = utils.dc if not coulomb else utils.dcoulomb
        self.updatable_marginals = np.array([marginal for marginal in range(num_marginals) if marginal not in skip_marginals])
        self.grad_func = grad_func
        self.explosion_scale = explosion_scale
        self.exploded = False
        self.dynthreshold = dynthreshold
        self.decreasing_couplings = []

        self.grads = [np.zeros((self.num_marginals, self.num_particles_per_marginal, self.dim)) for _ in range(2)]
        self.steps_history = [self.gd_step]
        self.deltafs = [0]
        self.gradnorm = []
        self.alphas = []
        self.dynindexes = []
        self.grads_history = []
        self.coupling_history = []
        self.max_iter_subsample = max_iter_subsample

        if self.threading:
            raise NotImplementedError('Threading is not supported.')
        if coulomb:
            print('(Solver) USING COULOMB COST !! Solver.dc has been modified.')
        if num_particles_per_marginal < 100 and threading:
            print('(Solver)  Threading is mostly unnecessary with few points per marginal.')
        if use_c and threading:
            raise ValueError('(Solver) Cannot use C code and threading together, remove an option.')

    
    def initialize_coupling(self, method, args):
        if method == 'uniform':
            self.initialize_coupling_uniform(*args)
        elif method == 'uniform2':
            self.initialize_coupling_uniform_many(*args)
        elif method == 'one-given':
            self.initialize_coupling_one_marginal_given(*args)
        elif method == 'from-coupling':
            self.coupling = args
        elif method == 'form-a':
            a = args
            self.coupling = np.zeros((self.num_marginals, self.num_particles_per_marginal, self.dim))
            
            for marginal in range(self.num_marginals):
                samples = stats.uniform.rvs(loc=a[marginal], scale=1, size=(self.num_particles_per_marginal, self.dim))
                self.coupling[marginal] = samples
        else:
            raise ValueError('Expected method in [uniform].')

        if self.adaptive_step_size:
            self.gradient_history = np.zeros(self.coupling.shape)
        
        self.initialized = True

    
    def initialize_coupling_one_marginal_given(self, min_bound, max_bound, marginal, samples):
        self.initialize_coupling_uniform(min_bound, max_bound)
        self.coupling[marginal] = samples

    
    def initialize_uniform_boxes(self, a):
        self.coupling = np.zeros((self.num_marginals, self.num_particles_per_marginal, self.dim))
        
        for marginal in range(self.num_marginals):
            samples = stats.uniform.rvs(loc=a[marginal], scale=1, size=self.num_particles_per_marginal)
            self.coupling = samples
        
    
    def initialize_coupling_uniform(self, min_bound, max_bound):
        spread = max_bound - min_bound
        entries_per_marginal = self.num_particles_per_marginal * self.dim
        self.coupling = np.zeros((self.num_marginals, self.num_particles_per_marginal, self.dim))

        for marginal in range(self.num_marginals):
            self.coupling[marginal] = stats.uniform.rvs(loc=min_bound, scale=spread, size=entries_per_marginal).reshape((1, self.num_particles_per_marginal, self.dim))
            self.coupling[marginal][:self.num_particles_per_marginal//2] = self.coupling[marginal][self.num_marginals//2:] ##
            pdb.set_trace()


    def initialize_coupling_uniform_many(self, min_bounds, max_bounds):
        entries_per_marginal = self.num_particles_per_marginal * self.dim
        self.coupling = np.zeros((self.num_marginals, self.num_particles_per_marginal, self.dim))
        x = np.array([stats.uniform.rvs(loc=0, scale=1.5, size=entries_per_marginal // 2),
            stats.uniform.rvs(loc=-3, scale=1, size=entries_per_marginal // 2)])
        self.coupling[0] = x.reshape((150, 1))
        self.coupling[1] = stats.uniform.rvs(loc=2, scale=2, size=entries_per_marginal).reshape((150, 1))


    def update(self):
        if not self.initialized:
            raise ValueError('Solver is not initialized. Call initialize_coupling before updates.')

        prev_coupling = self.coupling.copy()
        self.coupling = self.update_nothreading()

        self.compute_dynindex_coupling(prev_coupling)
        self.check_not_exploded()

        if (self.iteration % self.execution.save_samples_frequency == 0) or (self.iteration == self.execution.num_iterations_per_solver-1):
            self.coupling_history.append(self.coupling.copy())

        self.iteration += 1


    def check_not_exploded(self):
        max_point = self.coupling.max()

        if max_point >= self.explosion_scale:
            self.exploded = True
            self.explosion_value = max_point


    def compute_dynindex_coupling(self, prev_coupling):
        deltas = np.sqrt(np.sum(((self.coupling - prev_coupling)/self.gd_step)**2, axis=(0, 2)))
        self.dynindex = deltas.mean()
        self.dynindexes.append(self.dynindex)


    def update_nothreading(self):
        new_coupling = np.copy(self.coupling)
        self.grads[-2] = self.grads[-1].copy()
        subsample = self.iteration < self.max_iter_subsample if self.max_iter_subsample is not None else False

        for marginal in self.updatable_marginals:
            for particle in range(self.num_particles_per_marginal):
                c_grad = self.dc(self.coupling[:, particle], self.lambdas, convex=False)[marginal]
                k_grad = self.kernel.gradient(
                    self.coupling, 
                    marginal, 
                    particle, 
                    subsample=subsample,
                    subsample_fraction=self.kernel.subsample_fraction
                    ) * (self.alpha * self.lambdas[marginal])
                grad = -c_grad + k_grad

                self.grads[-1][marginal, particle] = grad

                new_coupling[marginal, particle] = new_coupling[marginal, particle] + grad*self.gd_step

        if (self.iteration % self.execution.save_samples_frequency == 0) or (self.iteration == self.execution.num_iterations_per_solver-1):
            self.grads_history.append(self.grads[-1])

        return new_coupling
