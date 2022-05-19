import pdb
import numpy as np
import utils


class SVGDKernel:
    def __init__(self, adaptive=False, potential_grads=None, bandwidth=1):
        self.adaptive = adaptive
        self.potential_grads = [] if potential_grads is None else potential_grads
        self.subsample_fraction = 1

        self.kernel = lambda x, y: np.exp(-0.5*bandwidth*utils.norm2(x - y))
        self.kernel_grad2 = lambda x, y: bandwidth * (x-y) * (np.exp(-0.5*bandwidth*utils.norm2(x - y)) + 1e-12)


    def copy(self):
        return SVGDKernel(adaptive=self.adaptive, potential_grads=self.potential_grads)

    
    def append_potential_gradient(self, potential_grad):
        self.potential_grads.append(potential_grad)

    
    def update(self, coupling):
        if self.adaptive:
            self.kernel, self.kernel_grad2 = utils.svgd_adaptive_gradient(coupling, p=2)

    
    def gradient(self, coupling, marginale, particle, subsample=False, subsample_fraction=0.25):
        _, num_particles_per_marginal, dim = coupling.shape
        grad = np.zeros(dim)
        X_part = coupling[marginale, particle]
        potential_grad = self.potential_grads[marginale]

        if subsample:
            indices = np.random.choice(
                np.arange(num_particles_per_marginal),
                replace=False,
                size=int(subsample_fraction*num_particles_per_marginal)
            )
        else:
            indices = range(num_particles_per_marginal)

        for other_particle in indices:
            X_other = coupling[marginale, other_particle]
            grad = grad - potential_grad(X_other)*self.kernel(X_part, X_other) + self.kernel_grad2(X_part, X_other)
        grad = grad / len(indices)

        if np.isnan(grad).any():
            raise ValueError('(SVGDKernel.gradient) Encountered infinite value:\n', grad)

        return grad
