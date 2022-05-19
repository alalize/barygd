import pdb
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import make_dir
import pickle
import numpy as np


class Execution:
    def __init__(self):
        self.iteration_records = []
        self.distances = {}
        self.average_distances = {}
        self.std_distances = {}
        self.reference_distances = {}
        self.std_reference_distances = {}
        self.start_iteration = 0
        self.iteration = 0
        self.solvers = []


    def save(self):
        folder = os.path.join('runs', self.name)
        make_dir(folder)
        logging.info('Created save folder \'{}\' .'.format(folder))

        with open(os.path.join(folder, '{}.pickle'.format(self.name)), 'wb') as file_stream:
        	simulation_data = {
        		'name': self.name,
        		'random-seed': self.random_seed,
        		'dimension': self.dimension,
        		'num-marginals': self.num_marginals,
        		'num-particles-per-marginal': self.num_particles_per_marginal,
        		'weights': self.weights,
        		'num-solvers': self.num_solvers,
        		'covariance-scale': self.covariance_scale,
        		'num-iterations-per-solver': self.num_iterations_per_solver,
        		'save-samples-frequency': self.save_samples_frequency,
        		'dynamics-threshold': self.dynamics_alpha_threshold,
        		'svgd-h0': self.initial_gd_step,
        		'svgd-alpha': self.initial_alpha,
        		'kernel-scale': self.kernel_scale,
        		'alpha-frequency': self.solver_alpha_freq,
        		'gaussian-barycenter-mean': self.gaussian_barycenter_mean,
        		'gaussian-barycenter-covariance': self.gaussian_barycenter_covariance,
        		'barycenters-records': self.barycenter_samples_records,
        		'final-couplings': [solver.coupling for solver in self.solvers],
                'distances': self.distances,
                'reference-distance-runs': self.reference_distance_runs,
                'iteration-records': self.iteration_records,
                'average-distances': self.average_distances,
                'std-distances': self.std_distances,
                'reference-distances': self.reference_distances,
                'std-reference-distances': self.std_reference_distances,
                'start-iteration': self.start_iteration,
                'marginal-means': self.means.tolist(),
                'marginal-covariances': self.covariances.tolist(),
                'iteration': self.iteration, ##
                'solvers-alphas': [self.solvers[i].alpha for i in range(self.num_solvers)],
                'solvers-steps': [self.solvers[i].gd_step for i in range(self.num_solvers)],
                'solvers-initial-steps': [self.solvers[i].initial_gd_step for i in range(self.num_solvers)],
                'solvers-dynthresholds': [self.solvers[i].dynthreshold for i in range(self.num_solvers)],
                'solvers-couplings': [self.solvers[i].coupling for i in range(self.num_solvers)],
                'coupling-history': [self.solvers[i].coupling_history for i in range(self.num_solvers)],
                'subsample-fraction': self.subsample_fraction,
                'max-subsampling-iteration': self.max_iter_subsample,
                'nongaussian': self.nongaussian
        	}
        	pickle.dump(simulation_data, file_stream)
        
        logging.info('[EXECUTION] Saved the state.')

    
    def load(self, file):
        with open(file, 'rb') as file_stream:
            simulation_data = pickle.load(file_stream)

            self.name = simulation_data['name']
            self.random_seed = simulation_data['random-seed']
            self.dimension = simulation_data['dimension']
            self.num_particles_per_marginal = simulation_data['num-particles-per-marginal']
            self.num_iterations_per_solver = simulation_data['num-iterations-per-solver']
            self.num_solvers = simulation_data['num-solvers']
            self.gaussian_barycenter_mean = simulation_data['gaussian-barycenter-mean']
            self.gaussian_barycenter_covariance = simulation_data['gaussian-barycenter-covariance']
            self.barycenter_samples_records = simulation_data['barycenters-records']
            self.distances = simulation_data['distances']
            self.reference_distance_runs = simulation_data['reference-distance-runs']
            self.iteration_records = simulation_data['iteration-records']
            self.average_distances = simulation_data['average-distances']
            self.reference_distances = simulation_data['reference-distances']
            self.start_iteration = simulation_data['start-iteration']
            self.iteration = simulation_data['iteration']
            self.means = np.array(simulation_data['marginal-means'])
            self.covariances = np.array(simulation_data['marginal-covariances'])
            self.iteration = simulation_data['iteration']
            for s in range(self.num_solvers):
                self.solvers[s].alpha = simulation_data['solvers-alphas'][s]
                self.solvers[s].gd_step = simulation_data['solvers-steps'][s]
                self.solvers[s].initial_gd_step = simulation_data['solvers-initial-steps'][s]
                self.solvers[s].dynthreshold = simulation_data['solvers-dynthresholds'][s]
                self.solvers[s].coupling = simulation_data['solvers-couplings'][s]
                self.solvers[s].coupling_history = np.array(simulation_data['coupling-history'][s])
            self.subsample_fraction = simulation_data['subsample-fraction']
            self.max_iter_subsample = simulation_data['max-subsampling-iteration']
            self.nongaussian = simulation_data['nongaussian']

        logging.info('Recovered state from file \'{}\' .'.format(file))


    def from_args(args):
        execution = Execution()

        execution.dimension = args.dimension
        execution.num_marginals = args.num_marginals
        execution.num_particles_per_marginal = args.num_particles_per_marginal
        execution.weights = (1/execution.num_marginals) * np.ones(execution.num_marginals)
        assert len(execution.weights) == execution.num_marginals, 'The length of the weights vector is different from the number of marginals.'

        execution.deltacoeff = args.deltacoeff
        execution.subsample_fraction = args.subsampling
        execution.max_iter_subsample = args.max_subsampling_iter

        execution.num_solvers = args.num_solvers
        execution.save_samples_frequency = args.save_samples_frequency
        execution.num_iterations_per_solver = args.num_iter

        execution.mean_scale = args.random_gaussian_barycenter_mean_scale
        execution.covariance_scale = args.random_gaussian_barycenter_covariance_scale

        execution.dynamics_alpha_threshold = args.dynscalth

        execution.initial_gd_step = args.initial_step
        execution.initial_alpha = args.initial_alpha
        execution.solver_alpha_freq = args.alpha_update_frequency
        execution.kernel_scale = args.kernel_scale
        execution.explosion_scale = args.explosion_scale
        execution.solver_kernel = args.solver_kernel

        execution.start_from_marginals = args.start_from_marginals
        execution.start_from_diagonal = args.start_from_diagonal
        execution.reference_distance_runs = args.reference_distance_runs
        execution.distance_modes = args.distances
        execution.random_seed = args.random_seed

        execution.nongaussian = args.non_gaussian_problem

        return execution


