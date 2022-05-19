import pdb
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.image
from utils import make_dir
import scipy


class Plotter:
    def __init__(self, execution, show):
        self.show = show
        self.execution = execution

        self.fig_width_cm = 30
        self.fig_height_cm = 24
        self.inOfcm = lambda cm: cm / 2.54

        logger = logging.getLogger('matplotlib')
        logger.setLevel(logging.INFO)
        logger.disabled = True


    def plot_and_save_steps(self, name):
        plt.title('steps history')
        plt.grid()
        plt.yscale('log')
        for s in range(self.execution.num_solvers):
            plt.plot(self.execution.solvers[s].steps_history, c='black', label=r'$h$')
            plt.plot(self.execution.solvers[s].deltafs, c='red', label=r'$1/L$')
            plt.plot(self.execution.solvers[s].alphas, c='green', label=r'$\alpha$')
            plt.plot(self.execution.solvers[s].dynindexes, c='gold', label=r'$\nabla F(x_n)$')
        plt.legend()

        make_dir(os.path.join('runs', name))
        path = os.path.join('runs', name, '{}-steps.png'.format(name))
        plt.savefig(path)
        logging.info('Saved h, 1/L, alpha, grad curves image to \'{}\' .'.format(path))

        if self.show:
            plt.show()


    def plot_and_save_distances(self, name, modes):
        colors = ['red', 'blue', 'green', 'black', 'gold']

        iteration_records = np.array(self.execution.iteration_records)

        fig, axis = plt.subplots(nrows=1, ncols=2)
        fig.suptitle(name)
        fig.set_size_inches(self.inOfcm(self.fig_width_cm), self.inOfcm(self.fig_height_cm))

        axis[1].set_title('log scale on ordinates')
        axis[1].set_yscale('log')
        axis[0].set_xlabel('iteration')
        axis[1].set_xlabel('iteration')
        axis[0].set_ylabel(r'$d^2(T {}_{\#} \gamma_N(t), T {}_{\#} \gamma^\star)$')
        axis[1].set_ylabel(r'$d^2(T {}_{\#} \gamma_N(t), T {}_{\#} \gamma^\star)$')
        axis[0].grid()
        axis[1].grid()

        x = iteration_records + 1

        for mi, mode in enumerate(modes):
            axis[0].plot(x, self.execution.average_distances[mode], label=mode, c=colors[mi])
            axis[1].plot(x, self.execution.average_distances[mode], label=mode, c=colors[mi])
            for s in range(self.execution.num_solvers):
                axis[0].plot(x, self.execution.distances[mode][s], c=colors[mi], alpha=0.1)
                axis[1].plot(x, self.execution.distances[mode][s], c=colors[mi], alpha=0.1)
                
            dist_mode_lower = self.execution.average_distances[mode] - 2*self.execution.std_distances[mode]
            dist_mode_upper = self.execution.average_distances[mode] + 2*self.execution.std_distances[mode]
            axis[0].fill_between(x, dist_mode_lower, dist_mode_upper, color=colors[mi], alpha=0.2)

            axis[0].hlines(y=self.execution.reference_distances[mode], xmin=x.min(), xmax=x.max(), linestyle='dashed', color=colors[mi]);
            axis[1].hlines(y=self.execution.reference_distances[mode], xmin=x.min(), xmax=x.max(), linestyle='dashed', color=colors[mi]);

            if mode != 'ratiovarw2':
                ref_lower = self.execution.reference_distances[mode] - 2*self.execution.std_reference_distances[mode]
                ref_upper = self.execution.reference_distances[mode] + 2*self.execution.std_reference_distances[mode]
                axis[0].fill_between(x, ref_lower, ref_upper, color=colors[mi], alpha=0.2)
                axis[0].hlines(y=ref_lower, xmin=x.min(), xmax=x.max(), color=colors[mi], alpha=0.2)
                axis[0].hlines(y=ref_upper, xmin=x.min(), xmax=x.max(), color=colors[mi], alpha=0.2)

        #axis[0].set_ylim([-1, 2])
        axis[0].legend()
        axis[1].legend()
        logging.info('Plotted distance curves.')

        make_dir(os.path.join('runs', name))
        path = os.path.join('runs', name, '{}.png'.format(name))
        plt.savefig(path)
        logging.info('Saved distance curves image to \'{}\' .'.format(path))

        if self.show:
            plt.show()

    
    def plot_and_save_2D_movie_grads(self, name):
        x0, y0 = np.linspace(-5, 15, 100), np.linspace(-5, 15, 100)
        x, y = np.meshgrid(x0, y0)
        grid = np.vstack([x.ravel(), y.ravel()])

        if self.execution.dimension <= 2:
            gaussian_barycenter_mean = self.execution.gaussian_barycenter_mean
            gaussian_barycenter_covariance = self.execution.gaussian_barycenter_covariance
        else:
            gaussian_barycenter_mean = self.execution.gaussian_barycenter_mean[:2]
            gaussian_barycenter_covariance = self.execution.gaussian_barycenter_covariance[:2, :2]

        gaussian_barycenter_pdf = lambda x, mean=gaussian_barycenter_mean, covariance=gaussian_barycenter_covariance: scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=covariance)
        gaussian_barycenter_density = np.array([gaussian_barycenter_pdf(p) for p in grid.T]).reshape(x.shape)

        fig, ax = plt.subplots()
        fig.suptitle(name)
        fig.set_size_inches(self.inOfcm(self.fig_width_cm), self.inOfcm(self.fig_height_cm))

        make_dir(os.path.join('runs', name))
        path = os.path.join('runs', name, '{}-movie.mp4'.format(name))

        writer = anim.FFMpegWriter(fps=15)
        writer.setup(fig=fig, outfile=path, dpi=100)
        plt.ion()

        d = self.execution.dimension if self.execution.dimension <= 2 else 2

        marginal_density = []
        for marginal in range(self.execution.num_marginals):
            mean = self.execution.means[marginal][:d]
            inv_cov = np.linalg.inv(self.execution.covariances[marginal][:d, :d])
            norm_cte = np.linalg.det(inv_cov)/(2*np.pi)
            def mdensity(x, y):
                p = np.array([x, y]) - mean
                val = np.exp(-0.5*p.dot(inv_cov.dot(p))) * norm_cte
                return val
            kde_density = mdensity(x0[:, None], y0[None, :]).T
            marginal_density.append(kde_density)

        for i in range(len(self.execution.barycenter_samples_records)):
            ax.clear()
            ax.grid()
            ax.set_title('iteration: {}'.format(self.execution.iteration_records[i]))

            # forced solver index 0; temporary
            barycenter_total_samples = np.vstack(self.execution.barycenter_samples_records[i][0][:, :d])
            kde = scipy.stats.gaussian_kde(barycenter_total_samples.T)
            kde_density = np.reshape(kde.pdf(grid).T, x.shape)

            ax.contourf(x, y, kde_density, cmap='Reds', levels=10, alpha=0.7)
            ax.contour(x, y, gaussian_barycenter_density, cmap='Greys', levels=10, alpha=1)
            ax.scatter(
                barycenter_total_samples[:, 0], 
                barycenter_total_samples[:, 1], 
                marker='o', 
                c='black', 
                alpha=0.8, 
                edgecolors='gold'
            )

            for marginal in range(self.execution.num_marginals):
                ax.contour(x, y, marginal_density[marginal], cmap='Blues', levels=10, alpha=1)

            for solver in range(self.execution.num_solvers):
                coupling = self.execution.solvers[solver].coupling_history[i]
                #grads = self.execution.solvers[solver].grads_history[i]
                step = self.execution.solvers[solver].steps_history[self.execution.iteration_records[i]]
                
                for marginal in range(self.execution.num_marginals):
                    #coupling_m, grads_m = coupling[marginal][:, :d], grads[marginal][:, :d]
                    coupling_m = coupling[marginal][:, :d]

                    kde_marginal = scipy.stats.gaussian_kde(coupling_m.T)
                    kde_marginal_density = np.reshape(kde_marginal.pdf(grid).T, x.shape)

                    ax.contour(x, y, kde_marginal_density, cmap='Blues', levels=10, alpha=0.2)
                    ax.scatter(
                        coupling_m[:, 0], 
                        coupling_m[:, 1], 
                        marker='o', 
                        c='blue',
                        alpha=0.3,
                        edgecolors='black'
                    )

                    #ax.quiver(
                    #    coupling_m[:, 0], 
                    #    coupling_m[:, 1], 
                    #    grads_m[:, 0], 
                    #    grads_m[:, 1],
                    #    alpha=0.2
                    #)
            
            writer.grab_frame()
            plt.pause(0.01)

        plt.ioff()
        plt.close()
        writer.finish()


    def plot_and_save_2D(self, name):
        x, y = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
        grid = np.vstack([x.ravel(), y.ravel()])

        barycenter_total_samples = np.vstack(self.execution.barycenter_samples_records[-1])
        kde = scipy.stats.gaussian_kde(barycenter_total_samples.T)
        kde_density = np.reshape(kde.pdf(grid).T, x.shape)
        gaussian_barycenter_pdf = lambda x, mean=self.execution.gaussian_barycenter_mean, covariance=self.execution.gaussian_barycenter_covariance: scipy.stats.multivariate_normal.pdf(x, mean=mean, cov=covariance)
        gaussian_barycenter_density = np.array([gaussian_barycenter_pdf(p) for p in grid.T]).reshape(x.shape)

        plt.grid()
        plt.title('dimension {} after {} iter., gdstep {}'.format(self.execution.dimension, self.execution.num_iterations_per_solver, self.execution.initial_gd_step))
        plt.contourf(x, y, kde_density, cmap='Reds', levels=10, alpha=0.8);
        plt.contourf(x, y, gaussian_barycenter_density, cmap='Greys', levels=10, alpha=0.4);
        plt.scatter(barycenter_total_samples[:, 0], barycenter_total_samples[:, 1], marker='o', c='black', alpha=0.3, edgecolors='gold');
        fig = plt.gcf()
        fig.suptitle(name)
        fig.set_size_inches(self.inOfcm(self.fig_width_cm), self.inOfcm(self.fig_height_cm))
        logging.info('Plotted 2D state.')

        make_dir(os.path.join('runs', name))
        path = os.path.join('runs', name, '{}-state.png'.format(name))
        plt.savefig(path)
        logging.info('Saved 2D state image to \'{}\' .'.format(path))

        if self.show:
            plt.show()
