import pdb
import subprocess
import numpy as np
from progress.bar import Bar


num_solvers = 5
num_particles_per_marginal = 50
mmd_kernel = 'id'
num_iterations_per_solver = 300

num_simulations = 1
#initial_steps = np.linspace(0.1, 1, num_simulations)
initial_steps = [0.1]
dimensions = [64]
num_marginals = [20]
#dimensions = np.array([2**(i+5) for i in range(1, num_simulations+1)])
#num_marginals = np.array([30 for i in range(1, num_simulations+1)])

num_total_simulations = len(initial_steps)*len(dimensions)*len(num_marginals)

#with open('names.txt', 'r') as file_stream:
#    names = [n[:-1] for n in file_stream.readlines()]
names = ['110322' for _ in range(len(initial_steps)*len(dimensions)*len(num_marginals))]

try:
    bar = Bar('SIM:', max=num_total_simulations, suffix='ETA %(eta)d sec, %(index)d/%(max)d - %(percent)d %%')
    for step_index, initial_step in enumerate(initial_steps):
        for dimension_index, dimension in enumerate(dimensions):
            for num_marg_index, num_marg in enumerate(num_marginals):
                subprocess.run(
                    ''.join([
                        'python3.9 ',
                        'sim_many_gaussians.py ',
                        '--mmd-kernel ',
                        mmd_kernel,
                        ' --dimension ',
                        str(dimension),
                        ' --num-marginals ',
                        str(num_marg),
                        ' --num-particles-per-marginal ',
                        str(num_particles_per_marginal),
                        ' --num-solvers ',
                        str(num_solvers),
                        ' --num-iterations-per-solver ',
                        str(num_iterations_per_solver),
                        ' --initial-step ',
                        str(initial_step),
                        ' --plot-distance ',
                        ' --plot-along ',
                        ' --dont-show ',
                        ' --name ',
                        names[bar.index]
                    ]), 
                    shell=True,
                    check=True
                )

                bar.next()
except subprocess.CalledProcessError as error:
    print('(LAUNCHER) Exitted at simulation {} ({}-{}-{})isdn due to exception:'.format(bar.index, initial_step, dimension, num_marg))
    print(str(error))
