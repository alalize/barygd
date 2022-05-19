import pdb
import os, sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from scipy.stats import gaussian_kde

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel import SVGDKernel
from solver import Solver
from mixture_bary_lp import sample_cdf
from qindics import unnormdensity_2disks_2d, potential_gradient_2disks_2d


dimension = 2
num_marginals = 2
num_particles_per_marginal = 250
weights = np.array([0.8, 0.2])
num_iterations = 300

svgd_h0 = 0.01
svgd_alpha = 10
svgd_damping = 1

laws = r'\mathcal{D}(2, 2, -2, -2) \,\to\, \mathcal{D}(2, -2, -2, 2)'

a, b, c, d = 2, 2, -2, -2
density_1 = unnormdensity_2disks_2d(a, b, c, d)
potential_1 = potential_gradient_2disks_2d(a, b, c, d)
potential_1 = lambda x, p=potential_1: p(x[0], x[1])

a, b, c, d = 2, -2, -2, 2
density_2 = unnormdensity_2disks_2d(a, b, c, d)
potential_2 = potential_gradient_2disks_2d(a, b, c, d)
potential_2 = lambda x, p=potential_2: p(x[0], x[1])

svgd_kernel = SVGDKernel()
svgd_kernel.append_potential_gradient(potential_1)
svgd_kernel.append_potential_gradient(potential_2)
svgd_solver = Solver(
	num_marginals=num_marginals,
	num_particles_per_marginal=num_particles_per_marginal,
	dim=dimension,
	gd_step=svgd_h0,
	lambdas=weights,
	kernel=svgd_kernel,
	alpha=svgd_alpha,
	alpha_freq=20,
	variable_alpha=False,
	adaptive_step_size=False,
	frozen_step=True,
	threading=False
)
svgd_solver.initialize_coupling('uniform', (-1, 1))


plt.ion()
fig, axis = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 8)

xoffset, height, delta_height = 0.1, 10, 0.5

x0, y0 = np.linspace(-5, 5, 200), np.linspace(-5, 5, 200)
x, y = np.meshgrid(x0, y0)
riem = ((5 - (-5)) / x0.size)**2

density1_background = density_1(x, y)
density1_background = density1_background / np.sum(np.sum(density1_background)*riem)
density2_background = density_2(x, y)
density2_background = density2_background / np.sum(np.sum(density2_background)*riem)

bar = Bar('GD:', max=num_iterations, suffix='ETA %(eta)d sec, %(index)d/%(max)d - %(percent)d %%')
for iteration in range(num_iterations):
	svgd_barycenter_samples = np.sum(np.array([weights[i]*svgd_solver.coupling[i, :] for i in range(num_marginals)]), axis=0)

	axis[0].clear()
	axis[0].set_ylim([0, height])
	axis[0].text(xoffset, height-3*delta_height, r'$N={} \quad T={}$'.format(num_particles_per_marginal, num_iterations))
	axis[0].text(xoffset, height-4*delta_height, r'$\Lambda = $' + str(weights))
	axis[0].text(xoffset, height-5*delta_height, '${}$'.format(laws))
	axis[0].text(xoffset, height-6*delta_height, r'$\alpha_t = ' + str(svgd_solver.alpha) + r'$')

	axis[1].clear()
	axis[1].grid()
	axis[1].contourf(x, y, density1_background, cmap='Blues', levels=10, alpha=0.2)
	axis[1].contourf(x, y, density2_background, cmap='Greens', levels=10, alpha=0.2)
	axis[1].scatter(svgd_barycenter_samples[:, 0], svgd_barycenter_samples[:, 1], marker='x', c='red')
	axis[1].set_xlim([-5, 5])
	axis[1].set_ylim([-5, 5])
	
	if iteration == num_iterations - 1:
		kde = gaussian_kde(svgd_barycenter_samples.T)
		points = np.vstack([x.ravel(), y.ravel()])
		kde_density = np.reshape(kde.pdf(points).T, x.shape)
		axis[1].contour(x, y, kde_density, cmap='Greys', levels=10, alpha=0.5)

	plt.pause(0.1)

	svgd_solver.update()
	svgd_solver.gd_step = svgd_h0 * (svgd_damping / (iteration+1))
	if iteration % svgd_solver.alpha_freq == 0:
		svgd_solver.alpha = svgd_alpha * (iteration + 1)

	bar.next()
bar.finish()

print('Saving figure...')
plt.savefig('img/{}_2disks_2d.png'.format(len(os.listdir('img/'))))

print('Saving pickle...')
with open('pickles/{}_2disks_2d.pickle'.format(len(os.listdir('pickles/'))), 'wb') as file_stream:
	samples = {
		'barycenter': svgd_barycenter_samples,
		'coupling': svgd_solver.coupling
	}
	pickle.dump(samples, file_stream)

print('Done.')
