import pdb
import numpy as np
import utils


class SVGDKernel:
    def __init__(self, adaptive=False, potential_grads=None, bandwidth=1):
        self.adaptive = adaptive
        self.potential_grads = [] if potential_grads is None else potential_grads

        self.kernel = lambda x, y: np.exp(-0.5*bandwidth*utils.norm2(x - y))
        self.kernel_grad2 = lambda x, y: bandwidth * (x-y) * (np.exp(-0.5*bandwidth*utils.norm2(x - y)) + 1e-12)


    def copy(self):
        return SVGDKernel(adaptive=self.adaptive, potential_grads=self.potential_grads)

    
    def append_potential_gradient(self, potential_grad):
        self.potential_grads.append(potential_grad)

    
    def update(self, coupling):
        if self.adaptive:
            self.kernel, self.kernel_grad2 = utils.svgd_adaptive_gradient(coupling, p=2)

    
    def gradient(self, coupling, marginal, particle):
        _, num_particles_per_marginal, dim = coupling.shape
        grad = np.zeros(dim)
        X_part = coupling[marginal, particle]
        potential_grad = self.potential_grads[marginal]

        for other_particle in range(num_particles_per_marginal):
            X_other = coupling[marginal, other_particle]
            grad = grad - potential_grad(X_other)*self.kernel(X_part, X_other) + self.kernel_grad2(X_part, X_other)
        grad = grad / num_particles_per_marginal

        if np.isnan(grad).any():
            raise ValueError('(SVGDKernel.gradient)  Infinite value encountered:\n', grad)

        return grad
