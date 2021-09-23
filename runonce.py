import sys
import shutil
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
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
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



np.random.seed(0)
now_str = str(datetime.datetime.now()).replace(' ', '_')
draw_freq = 50
w2_freq = 100

n_mesh = 256
mesh_x_min, mesh_x_max = -10, 10
mesh = np.linspace(mesh_x_min, mesh_x_max, num=n_mesh)

iss0 = 1e-11
bary_coords = np.array([0.5, 0.5])

# mode \in ['run', 'dependance-regularization', 'dependance-dimension', dependance-n-marginals', 'dependance-samples']
# also 'w2-iterations' with SVGD & LAWGD curves for all considered laws

simul_params = {
    'mode': 'dependance-iterations',
    'print-gradients': False,
    'check-converged': True,
    'convergence-threshold': 1e-4,
    'n_eigen': 50,
    'g_conv': False,
    'n_iterations': 10000,
    'initial_domain': [(-5, 5)],
    'fd_step_size': (mesh_x_max - mesh_x_min) / (n_mesh - 1),
    'gd_step_size': 1e-2,
    'initial-gd-step-size': iss0,
    'iss0': iss0,
    'min-bandwidth': 1,
    'adaptive-step-size': False,
    'adaptive-step-size-method': 'adagrad', # momentum or adagrad
    'momentum': 1,
    'adaptive-kernel': False,
    'kernel-order': 2,
    'regularization': 1e9,
    'samples-per-marginal': 250,
    'bary_coords': bary_coords.tolist(),
    'n_marginals': len(bary_coords),
    'laws': 'norms',
    'orthogonal-normals': True,
    'compute-w2': True,
    'track-w2': True,
    'track-integral': True,
    'equal-coupling-start': False,
    'keep-regularization-fixed': True,
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

print('Parameters:\n')
for k in simul_params.keys():
    print('{}: {}'.format(k, simul_params[k]))


from kernel import SVGDKernel
from solver import Solver

from svgd_context import get_context
_, grad_potentials, means, stds, istds, eta = get_context(simul_params)
svgd_kernel = SVGDKernel(potential_grads=grad_potentials)

solver = Solver(simul_params['n_marginals'], simul_params['samples-per-marginal'], simul_params['dimension'],\
    simul_params['initial-gd-step-size'], bary_coords, svgd_kernel, simul_params['regularization'], \
        adaptive_step_size=True, threading=True)
solver.initialize_coupling('uniform', (-5, 5))


import matplotlib.style as pltstyle
pltstyle.use('fast')

bar = Bar('BARYGD:', max=simul_params['n_iterations'], suffix='ETA %(eta)d sec, %(index)d/%(max)d - %(percent)d %%')
for iteration in range(simul_params['n_iterations']):
    solver.update()

    if iteration % 100 == 0:
        print('\r\n(Drawing)')
        plt.clf()
        plt.grid()
        plt.scatter(solver.coupling[0, :], solver.coupling[1, :], c='red', marker='+')
        plt.pause(0.5)
    bar.next()
bar.finish()

plt.clf()
plt.grid()
plt.scatter(solver.coupling[0, :], solver.coupling[1, :], c='gray', marker='+')
plt.show()
