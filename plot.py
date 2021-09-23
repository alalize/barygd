import pdb
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter


def plot_init(ion, simul_params):
    if ion: 
        plt.ion()

    if simul_params['dimension'] == 1:
        if simul_params['n_marginals'] == 2:
            ncols = 3 if simul_params['track-w2'] else 2
            fig, axs = plt.subplots(nrows=1, ncols=ncols)
        else:
            fig = plt.figure(figsize=plt.figaspect(2))
            ax0 = fig.add_subplot(2, 1, 1, projection='3d')
            ax1 = fig.add_subplot(2, 1, 2)
            axs = [ax0, ax1]
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(20, 16)
    fig.canvas.draw()
    cols = ['red', 'orange', 'purple']
    marks = ['o', '^', 'x']

    if simul_params['laws'] == '2d-sticks':
        x0 = np.linspace(-1, 4, 300)
        y0 = np.linspace(-1, 4, 300)
        X0, Y0 = np.meshgrid(x0, y0)

    context = {}
    context['fig'] = fig
    context['cols'] = cols
    context['marks'] = marks
    if simul_params['dimension'] == 1:
        context['axs'] = axs
    else:
        context['ax'] = ax
        if simul_params['laws'] == '2d-sticks':
            context['X0'] = X0
            context['Y0'] = Y0

    if simul_params['film']:
        #film_desc = { 'title':'BARYGD Sampling', 'artist':'Majabaut', 'comment':'LAWGD variant' }
        writer = FFMpegWriter(fps=15, metadata=simul_params['film_desc'])
        context['writer'] = writer

    return context   


def plot_intermediate(L, L_init, iteration, grads, context, simul_params, w2_dists=None, push=None, bary_density=None):
    bary_coords = np.array(simul_params['bary_coords'])

    fig = context['fig']
    cols = context['cols']
    marks = context['marks']
    if simul_params['dimension'] == 1:
        axs = context['axs']
    else:
        ax = context['ax']
        if simul_params['laws'] == '2d-sticks':
            X0, Y0 = context['X0'], context['Y0:']

    if simul_params['dimension'] == 1:
        axs[0].clear()
        axs[1].clear()
        if simul_params['track-w2']:
            axs[2].clear()

        if simul_params['rotate-view']:
            axs[0].view_init(30, iteration%360)
        axs[0].set_title(\
            'Iteration {}'.format(iteration) + r' -- Coupling $\gamma_{' + simul_params['algorithm'] + r'}$ in $\mathbf{R}^2$')
        axs[0].grid()
        if simul_params['n_marginals'] == 2:
            axs[0].scatter(L[0, :], L[1, :], marker='o', c='r', edgecolor='black')
            if push[1] is not None:
                axs[0].set_xlim(-7, 7)
                axs[0].plot(push[0], push[1](push[0]), c='gold', alpha=0.7, zorder=50, linewidth=2)
        elif simul_params['n_marginals'] == 3:
            axs[0].scatter(L[0, :], L[1, :], L[2, :], marker='o', c='r', edgecolor='black')

        axs[1].set_title(r'Points in $\mathbf{R}$' +  r' ($\alpha={:e}$, $h_t={:e}$)'.format(simul_params['regularization'], simul_params['gd_step_size']))
        axs[1].grid()
        for k in range(simul_params['n_marginals']):
            axs[1].hist(L[k, :], bins=50, density=True, color=cols[k], alpha=0.3)
            axs[1].scatter(L_init[k, :], np.zeros(L.shape[1]), c='black', marker='o', alpha=0.05)
            #axs[1].scatter(L[k, :], diffgrad[k, :], c=cols[k], marker=marks[k], alpha=0.5)
            axs[1].scatter(L[k, :], np.zeros(L[k, :].size), c=cols[k], marker=marks[k])
            axs[1].scatter(L[k, :], grads[k, :], c=cols[k], alpha=0.5, marker=marks[k], linewidth=2)
        if simul_params['laws'] == 'norm-arctan':
            axs[1].set_ylim(-1, 1)

        if simul_params['track-w2']:
            axs[2].set_title(r'$W_2$ distance over iterations')
            axs[2].grid()
            axs[2].plot(w2_dists[0], w2_dists[1], 'gray')
    else:
        ax.clear()
        ax.set_title(r'Iteration {} -- $(\alpha, h_t)$=({:>5e}, {:>5e})'.format(iteration, simul_params['regularization'], simul_params['gd_step_size']))
        #ax.set_xlim(-6, 6)
        #ax.set_ylim(-6, 6)
        ax.grid()
        ax.set_aspect('equal')

        mycolors = ['orange', 'blue', 'darkgreen']

        if not simul_params['equal-coupling-start'] or simul_params['equal-coupling-start'] and iteration > 10:
            samples = np.sum(np.array([L[ell]*bary_coords[ell] for ell in range(simul_params['n_marginals'])]), axis=0) # \sum_k \lambda_k X_t^{k, \scdot}
            sea.kdeplot(x=samples[:, 0], y=samples[:,1], fill=True, color='red', alpha=0.4, zorder=3)
            ax.scatter(samples[:, 0], samples[:, 1], c='gray', marker='o', edgecolor='black', zorder=4)

            for k in range(simul_params['n_marginals']):
                sea.kdeplot(x=L[k, :, 0], y=L[k, :, 1], fill=True, color=mycolors[k], alpha=0.3, zorder=1)
                ax.scatter(L[k, :, 0], L[k, :, 1], c=mycolors[k], marker='x', edgecolor='black', zorder=2)

                if bary_density is not None:
                    ax.scatter(bary_density[:, 0], bary_density[:, 1], c='gold', alpha=0.05, zorder=-1)

                if simul_params['laws'] == '2d-sticks':
                    plt.contour(X0, Y0, fat_stick(X0, Y0, A[k], B[k], 5, eta=eta), cmap='Dark2', levels=50, alpha=0.1)
    
    if simul_params['film']:
        context['writer'].grab_frame()

    #plt.pause(0.001)


def plot_free():
    plt.ioff()
    plt.close()

