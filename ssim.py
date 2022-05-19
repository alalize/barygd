import pdb
import argparse
import os, sys
import datetime
import copy
import pickle
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from progress.bar import Bar
import scipy.stats
import ot

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mmd_kernel import MMDKernel
from kernel import SVGDKernel
from solver import Solver
from utils import barycenter_of_gaussians


make_dir = lambda d: os.makedirs(d, exist_ok=True) if not os.path.exists(d) else None


mmd_kernel_choices = ['gaussian', 'id', 'var', 'idvar']
distance_mode_choices = ['w2', 'mmd', 'var', 'id', 'idvar']

MMD_KERNEL_GAUSSIAN = 0
MMD_KERNEL_MEANS = 1
MMD_KERNEL_COVARIANCES = 2

DISTANCE_MODE_WASSERSTEIN = 0
DISTANCE_MODE_MMD = 1
DISTANCE_MODE_COVARIANCE = 2


parser = argparse.ArgumentParser()
parser.add_argument('--resume-from', default='', type=str, help='Pickle to resume distance computations from.')
parser.add_argument('--plot-final', action='store_true', help='Plots the marginals and barycenters (true and estimated) in d=2. Requires --resume-from.')
parser.add_argument('--plot-distance', action='store_true', help='Plots the average Wasserstein distance curve to true barycenter clouds. Requires --resume-from.')
parser.add_argument('--distance-mode', type=str, default=distance_mode_choices[DISTANCE_MODE_MMD], choices=distance_mode_choices, help='Method for distance computations. Default: MMD. Choices: {}.'.format(distance_mode_choices))
parser.add_argument('--mmd-kernel', type=str, default=mmd_kernel_choices[MMD_KERNEL_MEANS], choices=mmd_kernel_choices, help='Kernel to compute MMD distance estimate. Default: Distance of means. Choices: {}.'.format(mmd_kernel_choices))
parser.add_argument('--dimension', type=int, default=2)
parser.add_argument('--num-marginals', type=int, default=2)
parser.add_argument('--num-particles-per-marginal', type=int, default=50)
parser.add_argument('--num-solvers', type=int, default=5)
parser.add_argument('--weights', type=int, nargs='+', default=[0.5, 0.5])
parser.add_argument('--save-samples-frequency', type=int, default=10)
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
parser.add_argument('--plot-along', action='store_true', help='If set, plots after optimization.')
parser.add_argument('--dont-show', action='store_true', help='If set, will save image but not show the plot.')
parser.add_argument('--start-from-marginals', action='store_true', help='Starts from a coupling with correct marginals.')
parser.add_argument('--solver-kernel', type=str, choices=['svgd', 'mmd'], default='svgd', help='The penalization kernel used by the algorithm for the optimization.')
parser.add_argument('--dynscal', action='store_true', help='If set, alpha is updated only when the dynamics halts.')
parser.add_argument('--dynscalth', type=float, default=0.001, help='The threshold below which the dynamics is deemed halted.')
parser.add_argument('--explosion-scale', type=float, default=20, help='If any coordinate of coupling samples overflows this, an explosion occurred.')
parser.add_argument('--plot-best', action='store_true', help='If set, plots the coupling with the smallest distance.')
parser.add_argument('--start-from-diagonal', action='store_true', help='If set, samples one marginal and copies it along the coupling.')
args = parser.parse_args()


random_seed = args.random_seed
np.random.seed(random_seed)

if args.start_from_diagonal or args.start_from_marginals:
	xor = lambda a, b: (a and (not b)) or ((not a) and b)
	assert xor(args.start_from_diagonal, args.start_from_marginals), 'Options --start-from-diagonal and --start-from-marginals are mutually exclusive.'
if args.plot_final:
	assert args.dimension == 2, 'Option --plot-final is only available when dimension=2.'
	assert args.resume_from != '' or (args.resume_from == '' and args.plot_along), 'Option --plot-final requires specifying --resume-from.'
if args.plot_distance:
	assert args.resume_from != '' or (args.resume_from == '' and args.plot_along), 'Option --plot-distance requires specifying --resume-from if --plot-along is not set.'
if args.technical_name == '':
	now = datetime.datetime.now()
	make_two_digits = lambda x: x if len(x) == 2 else '0'+x

	args.technical_name_withouttime = '{}{}{}-({}-{}-{}-{})nNds-({})w-({}-{}-{})isiadam-({}-{}-{})itsfaf-{}rs'.format(
		make_two_digits(str(now.day)),
		make_two_digits(str(now.month)),
		str(now.year)[-2:],
		args.num_marginals,
		args.num_particles_per_marginal,
		args.dimension,
		args.num_solvers,
		args.weights,
		args.initial_step,
		args.initial_alpha,
		args.damping,
		args.num_iter,
		args.save_samples_frequency,
		args.alpha_update_frequency,
		args.random_seed
	)

	args.technical_name = '{}{}{}{}{}-({}-{}-{}-{})nNds-({})w-({}-{}-{})isiadam-({}-{}-{})itsfaf-{}rs'.format(
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
		args.damping,
		args.num_iter,
		args.save_samples_frequency,
		args.alpha_update_frequency,
		args.random_seed
	)
if args.name == '':
	args.name = ('MMD-' if args.solver_kernel == 'mmd' else 'SVGD-') + args.technical_name_withouttime

print('')
print('Name:', args.name)
print('  Technical name:', args.technical_name)


if args.resume_from == '':
	random_pd_matrix = lambda scale, dim: (lambda a, d: 0.5*np.eye(d) + a.T.dot(a))(scale*(np.random.rand(dim, dim)-0.5), dim)


	def gaussian_potential_gradient(mean, sigma):
		inv_sigma = numpy.linalg.inv(sigma)
		pot = lambda x, mean=mean, inv_sigma=inv_sigma: inv_sigma.dot(x - mean)

		return pot


	dimension = args.dimension
	num_marginals = args.num_marginals
	num_particles_per_marginal = args.num_particles_per_marginal
	weights = (1/num_marginals) * np.ones(num_marginals)
	assert len(weights) == num_marginals, 'The length of the weights vector is different from the number of marginals.'

	num_solvers = args.num_solvers
	save_samples_frequency = args.save_samples_frequency
	num_iterations_per_solver = args.num_iter

	mean_scale = args.random_gaussian_barycenter_mean_scale
	covariance_scale = args.random_gaussian_barycenter_covariance_scale
	means = mean_scale*np.random.rand(num_marginals, dimension)
	covariances = np.array([random_pd_matrix(covariance_scale, dimension) for _ in range(num_marginals)])
	potential_gradients = [gaussian_potential_gradient(mean, covariance) for mean in means for covariance in covariances]

	gaussian_barycenter_mean, gaussian_barycenter_covariance, _ = barycenter_of_gaussians(means, covariances, weights)

	dynamics_alpha_scaling = args.dynscal
	dynamics_alpha_threshold = args.dynscalth

	initial_gd_step = args.initial_step
	initial_alpha = args.initial_alpha
	damping = args.damping
	solver_alpha_freq = args.alpha_update_frequency
	kernel_scale = args.kernel_scale

	print('Setting up solver...')
	if args.solver_kernel == 'svgd':
		kernel = SVGDKernel(bandwidth=kernel_scale)
		for gradient in potential_gradients:
			kernel.append_potential_gradient(gradient)
	else:
		print('Preparing marginal samples for MMD kernel')
		marginal_samples = np.array([np.random.multivariate_normal(means[i], covariances[i], size=num_particles_per_marginal) for i in range(num_marginals)])
		kernel = MMDKernel(marginal_samples, bandwidth=kernel_scale)

	print('Creating solver')
	model_solver = Solver(
		num_marginals=num_marginals,
		num_particles_per_marginal=num_particles_per_marginal,
		dim=dimension,
		gd_step=initial_gd_step,
		lambdas=weights,
		kernel=kernel,
		alpha=initial_alpha,
		alpha_freq=solver_alpha_freq,
		variable_alpha=False,
		adaptive_step_size=False,
		frozen_step=True,
		threading=False,
		explosion_scale=args.explosion_scale
	)

	print('Initializing coupling')
	if args.start_from_marginals or args.start_from_diagonal:
		initial_coupling = np.zeros((num_marginals, num_particles_per_marginal, dimension))

		if args.start_from_diagonal:
			sample = np.random.multivariate_normal(means[0], covariances[0], size=num_particles_per_marginal)

		for marginal in range(num_marginals):
			if args.start_from_marginals:
				marginal_samples = np.random.multivariate_normal(means[marginal], covariances[marginal], size=num_particles_per_marginal)
				initial_coupling[marginal] = marginal_samples
			else:
				initial_coupling[marginal] = sample
		
		model_solver.initialize_coupling('from-coupling', initial_coupling)
	else:
		model_solver.initialize_coupling('uniform', (-2, 2))
	print('Solver set.')

	make_samples_from_coupling = lambda coupling, weights=weights, num_marginals=num_marginals: np.sum(np.array([weights[i]*coupling[i, :] for i in range(num_marginals)]), axis=0)
	solvers = [copy.deepcopy(model_solver) for _ in range(num_solvers)]
	barycenter_samples = [None for _ in range(num_solvers)]

	iteration_records = [] # for these two lists...
	barycenter_samples_records = [] # ... as many as [num_solvers*num_iterations_per_solver / save_samples_frequency]


	print('Optimization ({} solvers with {}*{} particles in dimension {})...'.format(num_solvers, num_marginals, num_particles_per_marginal, dimension))

	bar = Bar('GD:', max=num_iterations_per_solver, suffix='ETA %(eta)d sec, %(index)d/%(max)d - %(percent)d %%')
	
	solver_decay = [1]*num_solvers
	dynamics_restarts = 0
	alpha_schedule = lambda t: t**3

	for iteration in range(num_iterations_per_solver):
		for solver_index in range(num_solvers):
			if solvers[solver_index].exploded:
				solvers[solver_index].gd_step *= 0.5
				solvers[solver_index].coupling *= 0.8
				solvers[solver_index].exploded = False

				if solvers[solver_index].gd_step < dynamics_alpha_threshold:
					print('\n(GD Solver {}) Dynamics threshold: {}'.format(solver_index, dynamics_alpha_threshold))
					dynamics_alpha_threshold *= 0.5

				print('\n(GD Solver {}) Explosion ({}). Halving step size ({}) and restarting from 0.8*coupling.'.format(solver_index, solvers[solver_index].explosion_value, solvers[solver_index].gd_step))

			barycenter_samples[solver_index] = make_samples_from_coupling(solvers[solver_index].coupling)

			solvers[solver_index].update()
			solvers[solver_index].gd_step = initial_gd_step * (damping / solver_decay[solver_index])

			if (not dynamics_alpha_scaling) and (iteration % model_solver.alpha_freq == 0):
				solvers[solver_index].alpha = initial_alpha * solver_decay[solver_index]
				
			if dynamics_alpha_scaling and (solvers[solver_index].dynindex_coupling < dynamics_alpha_threshold):
				dynamics_restarts += 1

				solvers[solver_index].alpha = initial_alpha * alpha_schedule(dynamics_restarts)
				solver_decay[solver_index] = 0

				print('\n(GD Solver {}) Dynamics restarted with alpha: {}'.format(solver_index, solvers[solver_index].alpha))

			solver_decay[solver_index] = solver_decay[solver_index] + 1

		if iteration % save_samples_frequency == 0 or iteration == num_iterations_per_solver-1:
			iteration_records.append(iteration)
			barycenter_samples_records.append(copy.deepcopy(barycenter_samples))

		bar.next()
	bar.finish()


	print('Saving pickle...')
	folder = os.path.join('pickles', args.name, args.technical_name) if args.name != args.technical_name else os.path.join('pickles', args.name)
	make_dir(folder)
	print('Created folder')
	with open(os.path.join(folder, '{}_{}many_gaussians_{}d_{}solvers_{}_{}samples_{}iter.pickle'.format(
		args.name, len(os.listdir(folder)), num_marginals, dimension, num_solvers, args.distance_mode, num_particles_per_marginal, num_iterations_per_solver
		)), 'wb') as file_stream:
		simulation_data = {
			'name': args.name,
			'random-seed': random_seed,
			'dimension': dimension,
			'num-marginals': num_marginals,
			'num-particles-per-marginal': num_particles_per_marginal,
			'weights': weights,
			'num-solvers': num_solvers,
			'covariance-scale': covariance_scale,
			'num-iterations-per-solver': num_iterations_per_solver,
			'save-samples-frequency': save_samples_frequency,
			'dynamics-threshold': dynamics_alpha_threshold,
			'svgd-h0': initial_gd_step,
			'svgd-alpha': initial_alpha,
			'kernel-scale': kernel_scale,
			'alpha-frequency': solver_alpha_freq,
			'svgd-damping': damping,
			'gaussian-barycenter-mean': gaussian_barycenter_mean,
			'gaussian-barycenter-covariance': gaussian_barycenter_covariance,
			'iterations-records': iteration_records,
			'barycenters-records': barycenter_samples_records,
			'final-couplings': [solver.coupling for solver in solvers]
		}
		pickle.dump(simulation_data, file_stream)


if args.resume_from != '':
	if args.resume_from == 'last':
		folder = os.path.join('pickles', args.name, args.technical_name) if args.name != args.technical_name else os.path.join('pickles', args.name)
		args.resume_from = os.path.join(folder, sorted(os.listdir(folder), key=lambda x: int(x.split(sep='_')[0]))[-1])


	with open(args.resume_from, 'rb') as file_stream:
		simulation_data = pickle.load(file_stream)

		args.name = simulation_data['name']
		dimension = simulation_data['dimension']
		num_particles_per_marginal = simulation_data['num-particles-per-marginal']
		num_iterations_per_solver = simulation_data['num-iterations-per-solver']
		num_solvers = simulation_data['num-solvers']
		iteration_records = simulation_data['iterations-records']
		gaussian_barycenter_mean = simulation_data['gaussian-barycenter-mean']
		gaussian_barycenter_covariance = simulation_data['gaussian-barycenter-covariance']
		barycenter_samples_records = simulation_data['barycenters-records']


if args.plot_distance:
	cov_from_vector = lambda u: u.reshape((u.size, 1)).dot(u.reshape(1, u.size))

	print('Computing {} distance estimate...'.format(args.distance_mode))
	empirical_solver_measure = np.ones((num_particles_per_marginal,)) / num_particles_per_marginal
	empirical_gaussian_measure = np.ones((num_particles_per_marginal,)) / num_particles_per_marginal
	distances_per_solver_iteration = np.zeros((num_solvers, len(iteration_records)))

	if args.distance_mode == 'mmd':
		if args.mmd_kernel == 'gaussian':
			kernel = lambda x, y, epsilon=0.1: np.exp(-(0.5/epsilon**2)*(x-y).T.dot(x-y))
		elif args.mmd_kernel == 'id':
			kernel = lambda x, y, dimension=dimension: x.reshape((1, x.size)).dot(y.reshape((y.size, 1))) / dimension
		elif args.mmd_kernel == 'var':
			kernel = lambda x, y, make_cov=cov_from_vector: np.trace(make_cov(x).dot(make_cov(y))) / dimension
		elif args.mmd_kernel == 'idvar':
			kernel = lambda x, y, make_cov=cov_from_vector: (x.reshape((1, x.size)).dot(y.reshape((y.size, 1))) + np.trace(make_cov(x).dot(make_cov(y)))) / dimension
		else:
			raise NotImplementedError('The variance kernel has not been implemented yet.')

	bar = Bar('DE:', max=num_solvers*len(iteration_records), suffix='ETA %(eta)d sec, %(index)d/%(max)d - %(percent)d %%')
	for solver_index in range(num_solvers):
		gaussian_barycenter_samples = np.random.multivariate_normal(gaussian_barycenter_mean, gaussian_barycenter_covariance, size=num_particles_per_marginal)

		for iteration_index in range(len(iteration_records)):
			solver_barycenter_samples = barycenter_samples_records[iteration_index][solver_index]

			if args.distance_mode == 'mmd':
				if args.mmd_kernel == 'var':
					solver_samples_copy = np.copy(solver_barycenter_samples)
					gaussian_samples_copy = np.copy(gaussian_barycenter_samples)

					solver_barycenter_samples = solver_barycenter_samples - np.mean(solver_barycenter_samples, axis=0)
					gaussian_barycenter_samples = gaussian_barycenter_samples - np.mean(gaussian_barycenter_samples, axis=0)

				distance = np.sqrt((1/num_particles_per_marginal**2) * (
						np.sum(np.sum([kernel(xi, xj) for xi in solver_barycenter_samples for xj in solver_barycenter_samples]))
						+ np.sum(np.sum([kernel(gi, gj) for gi in gaussian_barycenter_samples for gj in gaussian_barycenter_samples]))
						- 2*np.sum(np.sum([kernel(xi, gj) for xi in solver_barycenter_samples for gj in gaussian_barycenter_samples]))
					)
				)

				if args.mmd_kernel == 'var':
					solver_barycenter_samples = solver_samples_copy
					gaussian_barycenter_samples = gaussian_samples_copy
			else:
				if args.distance_mode in ['id', 'idvar']:
					mean_diff = np.mean(solver_barycenter_samples, axis=0) - gaussian_barycenter_mean
					id_distance = np.sqrt(mean_diff.reshape((1, dimension)).dot(mean_diff.reshape((dimension, 1))) / dimension)
				if args.distance_mode in ['var', 'idvar']:
					solver_samples_copy = np.copy(solver_barycenter_samples)
					solver_barycenter_samples = solver_barycenter_samples - np.mean(solver_barycenter_samples, axis=0)

					empirical_covariance = (1 / num_particles_per_marginal)*np.sum([cov_from_vector(xi) for xi in solver_barycenter_samples], axis=0)
					cov_diff = empirical_covariance - gaussian_barycenter_covariance
					cov_distance = np.sqrt(np.trace(cov_diff.T.dot(cov_diff))) / dimension

					solver_barycenter_samples = solver_samples_copy
				
				if args.distance_mode == 'id': distance = id_distance
				elif args.distance_mode == 'var': distance = cov_distance
				elif args.distance_mode == 'idvar': distance = id_distance + cov_distance

			distances_per_solver_iteration[solver_index, iteration_index] = distance

			bar.next()
	bar.finish()

	distances_average = np.sum(distances_per_solver_iteration, axis=0) / num_solvers
	distances_std = np.std(distances_per_solver_iteration, axis=0)
	distances_upper = distances_average + distances_std
	distances_lower = distances_average - distances_std

	print('Computing reference distance...')
	if args.distance_mode == 'mmd' or args.distance_mode in ['var', 'id', 'idvar']:
		if args.distance_mode in ['id', 'idvar'] or args.mmd_kernel == 'id':
			mean_diff = np.mean(gaussian_barycenter_samples, axis=0) - gaussian_barycenter_mean
			id_reference_distance = np.sqrt(mean_diff.reshape((1, dimension)).dot(mean_diff.reshape((dimension, 1))) / dimension)
		if args.distance_mode in ['var', 'idvar'] or args.mmd_kernel == 'var':
			samples_copy = np.copy(gaussian_barycenter_samples)
			gaussian_barycenter_samples = gaussian_barycenter_samples - np.mean(gaussian_barycenter_samples, axis=0)

			empirical_covariance = (1 / num_particles_per_marginal)*np.sum([cov_from_vector(xi) for xi in gaussian_barycenter_samples], axis=0)
			cov_diff = empirical_covariance - gaussian_barycenter_covariance
			cov_reference_distance = np.sqrt(np.trace(cov_diff.dot(cov_diff))) / dimension

			gaussian_barycenter_samples = samples_copy

		if args.distance_mode == 'mmd' and args.mmd_kernel == 'idvar':
			gbs2 = np.random.multivariate_normal(gaussian_barycenter_mean, gaussian_barycenter_covariance, size=num_particles_per_marginal)

			reference_distance = np.sqrt((1/num_particles_per_marginal**2) * (
					np.sum(np.sum([kernel(xi, xj) for xi in gbs2 for xj in gbs2]))
					+ np.sum(np.sum([kernel(gi, gj) for gi in gaussian_barycenter_samples for gj in gaussian_barycenter_samples]))
					- 2*np.sum(np.sum([kernel(xi, gj) for xi in gbs2 for gj in gaussian_barycenter_samples]))
				)
			)

		if args.distance_mode == 'id': reference_distance = id_reference_distance
		elif args.distance_mode == 'var': reference_distance = cov_reference_distance
		elif args.distance_mode == 'idvar': reference_distance = id_reference_distance + cov_reference_distance
	else:
		raise NotImplementedError('Reference distance is not implemented with for {} mode.'.format(args.distance_mode))


	print('Creating curves folder...')
	curve_folder = os.path.join('curves', args.name, args.technical_name) if args.name != args.technical_name else os.path.join('curves', args.name)
	make_dir(curve_folder)
	print('Created folder')
	print('Saving curves...')
	with open(os.path.join(curve_folder, '{}-{}.pickle'.format(args.name, len(os.listdir(curve_folder)))), 'wb') as file_stream:
		curves = {
			'distances-per-solver-iteration': distances_per_solver_iteration,
			'distances-average': distances_average,
			'distances-std': distances_std,
			'distances-upper-bound': distances_upper,
			'distances-lower-bound': distances_lower
		}
		pickle.dump(curves, file_stream)

	print('Creating image folder...')
	image_folder = os.path.join('img', args.name, args.technical_name) if args.name != args.technical_name else os.path.join('img', args.name)
	make_dir(image_folder)
	print('Created folder')

	print('Plotting curve...')
	iteration_records = np.array(iteration_records)
	fig, axis = plt.subplots()

	axis.set_title('dimension {} after {} iter., gdstep {}'.format(dimension, num_iterations_per_solver, initial_gd_step))
	axis.set_yscale('log')
	axis.grid()
	abscissa = iteration_records + 1
	axis.plot(abscissa, distances_average, c='black')
	axis.hlines(y=reference_distance, xmin=abscissa.min(), xmax=abscissa.max(), color='gray', linestyle='dashed')
	axis.fill_between(abscissa, distances_lower, distances_upper, color='gray', alpha=0.5)

	print('Saving curve...')
	plt.savefig(os.path.join(image_folder, '{}-{}.png'.format(args.name, len(os.listdir(image_folder)))))

	for solver_index in range(num_solvers):
		min_dist = distances_per_solver_iteration[solver_index].min()
		print('Minimum distance (solver {}): {}'.format(solver_index, min_dist))
		print('Reference distance:', reference_distance)

	if not args.dont_show:
		plt.show()


if args.plot_final:
	if args.plot_best:
		best_iteration = np.argmin(distances_per_solver_iteration[0])
	else:
		best_iteration = -1
	barycenter_total_samples = np.vstack(barycenter_samples_records[best_iteration])

	x, y = np.meshgrid(np.linspace(-2, 15, 100), np.linspace(-2, 15, 100))
	grid = np.vstack([x.ravel(), y.ravel()])

	kde = scipy.stats.gaussian_kde(barycenter_total_samples.T)
	kde_density = np.reshape(kde.pdf(grid).T, x.shape)
	gaussian_barycenter_pdf = lambda x, mean=gaussian_barycenter_mean, covariance=gaussian_barycenter_covariance: scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=covariance)
	gaussian_barycenter_density = np.array([gaussian_barycenter_pdf(p) for p in grid.T]).reshape(x.shape)

	plt.grid()
	plt.title('dimension {} after {} iter., gdstep {}'.format(dimension, num_iterations_per_solver, initial_gd_step))
	plt.contourf(x, y, kde_density, cmap='Reds', levels=10, alpha=0.8)
	plt.contourf(x, y, gaussian_barycenter_density, cmap='Greys', levels=10, alpha=0.4)
	plt.scatter(barycenter_total_samples[:, 0], barycenter_total_samples[:, 1], marker='o', c='black', alpha=0.3, edgecolors='gold')
	plt.show()


print('Done.')
