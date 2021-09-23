from os import stat
import pdb
import numpy as np
from numpy.lib.function_base import gradient
from numpy.lib.type_check import _nan_to_num_dispatcher
import utils
import scipy.stats as stats
from pathos.multiprocessing import ProcessPool
from psutil import cpu_count


class Solver:
    def __init__(self, num_marginals, num_particles_per_marginal, dim, gd_step,\
        lambdas, kernel, alpha, adaptive_step_size=False, threading=False, variable_alpha=False, alpha_freq=500,\
            frozen_step=False, coulomb=False):
        self.num_marginals = num_marginals
        self.num_particles_per_marginal = num_particles_per_marginal
        self.dim = dim
        self.gd_step = gd_step
        self.lambdas = lambdas
        self.kernel = kernel
        self.alpha = alpha
        self.variable_alpha = variable_alpha
        self.coupling = None
        self.adaptive_step_size = adaptive_step_size
        self.threading = threading
        self.initialized = False
        self.iteration = 0
        self.dynamics_measure = 0
        self.alpha_freq = alpha_freq
        self.frozen_step = frozen_step
        self.coulomb = coulomb
        
        self.dc = utils.dc if not coulomb else utils.dcoulomb

        if num_particles_per_marginal < 100 and threading:
            print('(Solver)  Threading is mostly unnecessary with few points per marginal.')

    
    def initialize_coupling(self, method, args):
        if method == 'uniform':
            self.initialize_coupling_uniform(*args)
        elif method == 'uniform2':
            self.initialize_coupling_uniform_many(*args)
        else:
            raise ValueError('Expected method in [uniform].')

        if self.adaptive_step_size:
            self.gradient_history = np.zeros(self.coupling.shape)
        
        self.initialized = True

    
    def initialize_coupling_uniform(self, min_bound, max_bound):
        spread = max_bound - min_bound
        entries_per_marginal = self.num_particles_per_marginal * self.dim
        self.coupling = np.zeros((self.num_marginals, self.num_particles_per_marginal, self.dim))

        for marginal in range(self.num_marginals):
            self.coupling[marginal] = stats.uniform.rvs(loc=min_bound, scale=spread, size=entries_per_marginal).reshape((1, self.num_particles_per_marginal, self.dim))


    def initialize_coupling_uniform_many(self, min_bounds, max_bounds):
        entries_per_marginal = self.num_particles_per_marginal * self.dim
        self.coupling = np.zeros((self.num_marginals, self.num_particles_per_marginal, self.dim))
        x = np.array([stats.uniform.rvs(loc=0, scale=1.5, size=entries_per_marginal // 2),
            stats.uniform.rvs(loc=-3, scale=1, size=entries_per_marginal // 2)])
        self.coupling[0] = x.reshape((150, 1))
        self.coupling[1] = stats.uniform.rvs(loc=2, scale=2, size=entries_per_marginal).reshape((150, 1))

        #self.coupling = np.zeros((self.num_marginals, self.num_particles_per_marginal, self.dim))

        #for marginal in range(self.num_marginals):
        #    min_bound, max_bound = min_bounds[marginal], max_bounds[marginal]
        #    spread = max_bound - min_bound

        #    self.coupling[marginal] = stats.uniform.rvs(loc=min_bound, scale=spread, size=entries_per_marginal).reshape((1, self.num_particles_per_marginal, self.dim))


    def update(self):
        if not self.initialized:
            raise ValueError('Solver is not initialized. Call initialize_coupling before updates.')

        self.iteration += 1
        self.kernel.update(self.coupling)

        if self.variable_alpha:
            self.update_alpha()

        prev_coupling = self.coupling.copy()

        if self.threading:
            self.coupling = self.update_threaded()
        else:
            self.coupling = self.update_nothreading()

        self.dynamics_measure = np.max(np.abs((self.coupling - prev_coupling) / prev_coupling))

    
    def update_alpha(self):
        def update():
            self.alpha *= 2
            if not self.frozen_step:
                self.gd_step /= 2

        if self.coupling.shape[-1] == 1:
            x_sort = np.argsort(self.coupling[0, :], axis=0)
            y_sort = np.argsort(self.coupling[1, :], axis=0)
            num_sort_defects = np.sum(list(map(Solver.bool2int, np.isclose(x_sort, y_sort))))

            if (num_sort_defects < 0.05*x_sort.size and self.dynamics_measure < 0.3) or self.iteration % self.alpha_freq == 0:
                update()
        else:
            if self.iteration % self.alpha_freq == 0 and self.iteration > 1:
                update()
    

    def bool2int(b):
        return 1 if b else 0


    def update_nothreading(self):
        new_coupling = np.copy(self.coupling)

        for marginal in range(self.num_marginals):
            for particle in range(self.num_particles_per_marginal):
                c_grad = self.dc(self.coupling[:, particle], self.lambdas, convex=False)[marginal] * self.gd_step
                k_grad = self.kernel.gradient(self.coupling, marginal, particle) * (self.gd_step * self.alpha * self.lambdas[marginal])

                if self.adaptive_step_size:
                    c_grad, k_grad, self.gradient_history = Solver.adagrad_step(self.gradient_history, c_grad, k_grad, marginal, particle)

                new_coupling[marginal, particle] = new_coupling[marginal, particle] - c_grad + k_grad

        return new_coupling

    
    def update_threaded(self):
        new_coupling = np.copy(self.coupling)

        results = []
        available_cpus = cpu_count(logical=False)

        with ProcessPool(nodes=available_cpus) as pool:
            total_num_points = self.num_marginals * self.num_particles_per_marginal
            batch_size, remainder = total_num_points // available_cpus, total_num_points % available_cpus
            batches = [np.arange(batch_size) + batch_size*i for i in range(available_cpus)]
            if remainder > 0:
                batches.append(np.arange(remainder) + batch_size*available_cpus)
            if self.iteration == 1:
                print('\r\n(Processing)  Available CPUs: {}   Batch size: {} ({})'.format(available_cpus, batch_size, remainder))

            for batch in batches:
                gradient_history = self.gradient_history.copy() if self.adaptive_step_size else None
                res = pool.apipe(Solver.threaded_gradient_step, *(batch, self.coupling.copy(), self.alpha, self.lambdas, self.gd_step, self.num_particles_per_marginal, self.kernel.copy(), gradient_history))
                results.append(res)
            for res in results:
                new_batch, marginals, particles, gradient_history = res.get()
                new_coupling[marginals, particles] = new_batch
                if self.adaptive_step_size:
                    self.gradient_history[marginals, particles] = gradient_history[marginals, particles]

        return new_coupling
    

    def threaded_gradient_step(batch, coupling, alpha, lambdas, gd_step, num_particles_per_marginal, kernel, gradient_history=None):
        new_coupling = np.zeros((len(batch), coupling.shape[-1]))
        marginals, particles = [], []

        for i, particle_index in enumerate(batch):
            marginal = particle_index // num_particles_per_marginal
            particle = particle_index % num_particles_per_marginal

            c_grad = utils.dc(coupling[:, particle], lambdas, convex=False)[marginal] * gd_step
            k_grad = kernel.gradient(coupling, marginal, particle) * (gd_step * alpha * lambdas[marginal])

            if gradient_history is not None:
                c_grad, k_grad, gradient_history = Solver.adagrad_step(gradient_history, c_grad, k_grad, marginal, particle)

            new_coupling[i] = coupling[marginal, particle] - c_grad + k_grad
            marginals.append(marginal)
            particles.append(particle)
    
        return new_coupling, marginals, particles, gradient_history


    def adagrad_step(gradient_history, c_grad, k_grad, marginal, particle):
        gradient_history[marginal, particle] += (-c_grad + k_grad)**2
        adafactor = 1/np.sqrt(gradient_history[marginal, particle] + 1e-8)
        c_grad, k_grad = c_grad*adafactor, k_grad*adafactor
        
        return c_grad, k_grad, gradient_history
