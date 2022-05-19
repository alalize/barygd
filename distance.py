import pdb
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import norm2
import numpy as np
import scipy
from numpy.linalg import eigvalsh


class Distance:
    def __init__(self, execution, modes):
        self.modes = modes
        self.execution = execution

        if self.execution.distances == {}:
            for mode in modes:
                self.execution.distances[mode] = [[] for _ in range(self.execution.num_solvers)]
            self.compute_reference_distances_and_q95()
        else:
            for ki, k in enumerate(self.execution.distances.keys()):
                if k != modes[ki]:
                    logging.warning('Distance incompatibility: mode {} is \'{}\' but state is \'{}\'.'.format(ki, modes[ki], k))


    def compute_average_curves(self):
        self.execution.average_distances = {}

        for mode in self.modes:
            dists = np.array(self.execution.distances[mode])

            mean_dist = np.mean(dists, axis=0)
            std = np.quantile(dists, 0.95, axis=0)/2

            self.execution.average_distances[mode] = mean_dist
            self.execution.std_distances[mode] = std


    def compute_reference_distances_and_q95(self):
        from run import Run

        self.execution.reference_distances = {}
        self.mean_bary_sigma = np.diag(self.execution.gaussian_barycenter_covariance).mean()

        for mode in self.modes:
            dists = []

            for _ in range(self.execution.reference_distance_runs):
                x = np.random.multivariate_normal(
                    self.execution.gaussian_barycenter_mean, 
                    self.execution.gaussian_barycenter_covariance, 
                    size=self.execution.num_particles_per_marginal)

                dists.append(self.distance_at_mode(mode, x))
            
            dist = np.mean(dists)
            q95 = np.quantile(dists, 0.95)/2

            self.execution.reference_distances[mode] = dist
            self.execution.std_reference_distances[mode] = q95

            logging.info(
                '[REF. DIST.] Reference for {}: {} ± {} (95% quantile). Up: {}'.format(
                    mode, dist, q95, dist + q95
                )
            )


    def step(self):
        for mode in self.modes:
            for s in range(self.execution.num_solvers):
                x = self.execution.barycenter_samples_records[-1][s]
                dist = self.distance_at_mode(mode, x)

                self.execution.distances[mode][s].append(dist)
                logging.info('\r\nDistance ({}) for Solver {}: {} at iter. {} .'.format(mode, s, dist, self.execution.iteration))


    def distance_at_mode(self, mode, x):
        if mode == 'id':
            dist = self.distance_id(x)
        elif mode == 'id2':
            dist = self.distance_id_sq(x)
        elif mode == 'var':
            dist = self.distance_cov_sq(x)
        elif mode == 'idvar':
            dist = self.distance_id_sq(x) + self.distance_cov_sq(x)
        elif mode == 'w2ansatz':
            dist = self.distance_ansatz_gaussian_sq(x)
        elif mode == 'wvar':
            dist = self.distance_variance(x)
        elif mode == 'wovar':
            dist = self.distance_wovar(x)
        elif mode == 'norm2':
            dist = self.distance_norm2(x)
        elif mode == 'ratiovarw2':
            dist = self.distance_ratio_varw2ansatz(x)
        elif mode == 'sigma':
            dist = np.diag(self.execution.gaussian_barycenter_covariance).min()
        else:
            raise ValueError('Unkown distance mode \'{}\' .'.format(mode))

        return dist


    def distance_id(self, x):
        return Distance.distance_id_(x, self.execution.gaussian_barycenter_mean, self.execution.dimension)


    def distance_id_(x, mean_ref, dimension):
        mean = np.mean(x, axis=0)
        dist = np.sqrt(norm2(mean - mean_ref) / dimension)

        return dist


    def distance_id_sq(self, x):
        return Distance.distance_id_sq_(
            x, 
            self.execution.gaussian_barycenter_mean, 
            self.execution.dimension
        )


    def distance_id_sq_(x, mean_ref, dimension):
        mean = np.mean(x, axis=0)
        dist = norm2(mean - mean_ref) / dimension

        return dist


    def distance_cov_sq(self, x):
        return Distance.distance_cov_sq_(
            x, 
            self.execution.gaussian_barycenter_covariance, 
            self.execution.dimension
        )

    
    def distance_cov_sq_(x, cov_ref, dimension):
        x_mean = np.mean(x, axis=0)
        x_cov = Distance.cov_from_samples(x, dimension)

        dist = np.trace((x_cov - cov_ref).dot(x_cov - cov_ref)) / dimension**2
        return dist


    def distance_norm2(self, x):
        return Distance.distance_norm2_(
            x,
            self.execution.gaussian_barycenter_covariance,
            self.execution.dimension
        )


    def distance_norm2_(x, cov_ref, dimension):
        x_mean = np.mean(x, axis=0)
        x_cov = Distance.cov_from_samples(x, dimension)
        spectral_radius = np.max(np.abs(eigvalsh(x_cov - cov_ref)))

        return spectral_radius


    def distance_ratio_varw2ansatz(self, x):
        return Distance.distance_ratio_varw2ansatz_(
            x, 
            self.execution.gaussian_barycenter_mean, 
            self.execution.gaussian_barycenter_covariance, 
            self.execution.dimension
        )


    def distance_ratio_varw2ansatz_(x, mean_ref, cov_ref, dimension):
        var = Distance.distance_cov_sq_(x, cov_ref, dimension)
        w2 = Distance.distance_wovar_(x, mean_ref, cov_ref)
        ratio = var / w2

        return ratio


    def distance_ansatz_gaussian_sq(self, x):
        return Distance.distance_ansatz_gaussian_sq_(
            x, 
            self.execution.gaussian_barycenter_mean, 
            self.execution.gaussian_barycenter_covariance
        )

    
    def distance_ansatz_gaussian_sq_(x, mean_ref, cov_ref):
        x_mean = np.mean(x, axis=0)
        x_cov = Distance.cov_from_samples(x, dimension=x.shape[1])

        dist = Distance.distance_wasserstein_gaussians(x_mean, x_cov, mean_ref, cov_ref)
        return dist


    def distance_wovar(self, x):
        return Distance.distance_wovar_(
            x, 
            self.execution.gaussian_barycenter_mean, 
            self.execution.gaussian_barycenter_covariance
        )


    def distance_wovar_(x, mean_ref, cov_ref):
        x_mean = np.mean(x, axis=0)
        x_cov = Distance.cov_from_samples(x, dimension=x.shape[1])

        _, dist = Distance.distance_wasserstein_gaussians_components(x_mean, x_cov, mean_ref, cov_ref)
        return dist

    
    def distance_variance(self, x):
        dist = Distance.distance_variance_(
            x, 
            self.execution.weights, 
            self.execution.means,
            self.execution.covariances, 
            self.execution.gaussian_barycenter_mean, 
            self.execution.gaussian_barycenter_covariance
        )
        return dist

    
    def distance_variance_(x, weights, means_ref, covs_ref, bmean_ref, bcov_ref):
        num_marginals, dimension = len(weights), x.shape[-1]

        x_mean = np.mean(x, axis=0)
        x_cov = Distance.cov_from_samples(x, dimension)
        
        var = 0
        for marginal in range(num_marginals):
            var_marg = Distance.distance_wasserstein_gaussians(means_ref[marginal], covs_ref[marginal], x_mean, x_cov)
            var_bary = Distance.distance_wasserstein_gaussians(means_ref[marginal], covs_ref[marginal], bmean_ref, bcov_ref)
            var = var + weights[marginal]*(var_marg - var_bary)
        
        return var


    def distance_wasserstein_gaussians(mean1, cov1, mean2, cov2):
        m, c = Distance.distance_wasserstein_gaussians_components(mean1, cov1, mean2, cov2)

        return m+c


    def distance_wasserstein_gaussians_components(mean1, cov1, mean2, cov2):
        dimension = mean1.size
        mean_dist_sq = norm2(mean1 - mean2)

        # tr(S_1 + S_2 - 2\sqrt{\sqrt{S_1} S_2 \sqrt{S_1}})
        sqrtCov1 = np.real_if_close(scipy.linalg.sqrtm(cov1))
        covProd = np.real_if_close(scipy.linalg.sqrtm(sqrtCov1.dot(cov2).dot(sqrtCov1)))
        cov_dist_sq = np.trace(cov1 + cov2 - 2*covProd)

        return mean_dist_sq/dimension, cov_dist_sq/dimension**2


    def cov_from_samples(x, dimension, epsilon=1e-12):
        num_particles = x.shape[0]
        x_mean = np.mean(x, axis=0)
        x_center = x - x_mean
        x_cov = np.eye(dimension)*epsilon

        for n in range(num_particles):
            col = x_center[n].reshape((dimension, 1))
            x_cov = x_cov + col.dot(col.T)
        x_cov = x_cov / (num_particles - 1)

        return x_cov
