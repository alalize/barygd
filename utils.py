import pdb
import numpy as np
import scipy.stats as stats 
import scipy.sparse as sparse
from scipy.linalg import sqrtm
from numpy.linalg import eigh
from skimage import color


def norm2(x):
    if x.size == 1:
        return x**2
    return x.dot(x)
norm = lambda x: np.sqrt(norm2(x))


def c(x, lambdas):
    def T(x):
        return np.sum(np.array([lambdas[i]*x[i] for i in range(x.shape[0])]), axis=0)
    cval = np.sum([lambdas[i]*norm2(x[i] - T(x)) for i in range(x.shape[0])], axis=0)

    return cval


def dc(x, lambdas, convex=False):
    dx = np.zeros(x.shape)
    conv = 1 if convex else 0
    for k in range(x.shape[0]):
        dx[k] = 2*lambdas[k]*(x[k] - np.sum([lambdas[i]*x[i] for i in range(0, x.shape[0])])) \
            + x[k]*conv
    return dx


def dcoulomb(x, lambdas, convex=False):
    dx = np.zeros(x.shape)
    for k in range(x.shape[0]):
        dx[k] = -(1/x.shape[0])*np.sum([1/norm2(x[k] - x[j]) for j in range(k+1, x.shape[0])])
    return dx


def dc_fd(x, lambds, h=1e-8):
    pass


def intc(coupling, lambdas):
    integral, num_samples = 0, coupling.shape[1]

    for i in range(num_samples):
        integral = integral + c(coupling[:, i], lambdas).item()
    integral = integral / num_samples

    return integral


def opt_intc_norms_1d(means, stds, lambdas):
    sum_means = sum([lambdas[i]*means[i] for i in range(len(means))])
    sum_stds = sum([lambdas[i]*stds[i] for i in range(len(stds))])
    integral = sum([lambdas[i]*((means[i] - sum_means)**2 + stds[i]**2 + sum_stds**2 - 2*stds[i]*sum_stds) for i in range(len(means))])

    return integral.item()


def opt_intc_norms(means, stds, lambdas):
    bary_mean, bary_std, _ = barycenter_of_gaussians(means, stds, lambdas)

    def sqw2_distance_norms(mean_1, mean_2, std_1, std_2):
        w2 = norm2(mean_1 - mean_2) + np.trace(std_1 + std_2 - 2*sqrtm(std_1.dot(std_2)))
        return w2
    
    integral = sum([li * sqw2_distance_norms(mi, bary_mean, si, bary_std) for (li, mi, si) in zip(lambdas, means, stds)])
    return integral.item()


def barycenter_of_gaussians(means, stds, lambdas, threshold=1e-9, maxiter=1000):
    dim = means[0].size

    bary_mean = sum([lambda_i*mean_i for (lambda_i, mean_i) in zip(lambdas, means)])

    def fixed_point_func(s):
        v = sum([li*sqrtm(sqrtm(s).dot(si).dot(sqrtm(s))) for (li, si) in zip(lambdas, stds)])
        return v

    thresholded_out = False
    bary_std = np.eye(dim)
    for i in range(maxiter):
        old, bary_std = bary_std, fixed_point_func(bary_std)

        if np.max(np.abs(bary_std - old)) < threshold:
            thresholded_out = True
            print('(barycenter_of_gaussians)  Thresholded out after {} iterations'.format(i))
            break
    
    return bary_mean, bary_std, thresholded_out


def svgd_gradient(K, dK,  dV, L, N, i):
    grad = np.zeros(L.shape[-1])
    for k in range(N):
        grad = grad - dV(L[k]) * K(L[i], L[k]) + dK(L[i], L[k])
    return grad / N


def svgd_adaptive_gradient(L, p=2, minband=1):
    n_points = L.shape[0]
    L_flat = np.reshape(L, (n_points, -1))

    distances = np.sqrt([norm2(L_flat[i] - L_flat[j]) for i in range(n_points) for j in range(i+1, n_points)])
    bandwidth = max(np.median(distances)**2 / np.log(n_points), minband)

    if p == 2:
        kernel = lambda x, y: np.exp(-norm2(x-y) / bandwidth)
        dkernel = lambda x, y: (2/bandwidth)*(x-y)*np.exp(-norm2(x-y) / bandwidth)
    else:
        kernel = lambda x, y: np.exp(-norm(x-y)**p / bandwidth)
        def dkernel(x, y):
            if norm(x - y) > 1e-10:
                return np.exp(-norm(x-y)**p / bandwidth) * p*norm(x-y)**(p-2) * (x-y)
            return np.zeros_like(x)

    return kernel, dkernel


def laws_2norm_family(means, stds, bary_weight1, simul_params):
    origin = lambda x: np.exp(-(x - means[0])**2/(2*stds[0]**2)) / np.sqrt(2*np.pi*stds[0]**2)

    if simul_params['laws'] == 'norm-arctan':
        push = lambda t: np.arctan(t)
        pushforward = lambda t: (1 + np.tan(t)**2) * np.exp(-np.tan(t)**2/2) / np.sqrt(2*np.pi)
    elif simul_params['laws'] == 'norm-exp':
        push = lambda t: np.exp(t)
        pushforward = lambda t: np.exp(-(np.log(t)-means[0])**2/(2*std_1**2)) / (t*np.sqrt(2*np.pi)*stds[0])
    else:
        push = lambda t: t+  means[1] - means[0]
        pushforward = lambda t: np.exp(-(t-means[1])**2/(2*stds[1]**2)) / np.sqrt(2*np.pi*stds[1]**2)

    normal_samples = stats.norm.rvs(loc=means[0], scale=stds[0], size=50000)
    if simul_params['laws'] == 'norm-arctan':
        normal_samples = bary_weight1*normal_samples + (1-bary_weight1)*np.arctan(normal_samples)
    elif simul_params['laws'] == 'norm-exp':
        normal_samples = bary_weight1*normal_samples + (1-bary_weight1)*np.exp(normal_samples)
    else:
        normal_samples = bary_weight1*normal_samples + (1-bary_weight1)*(normal_samples - means[0] + means[1])
    bary_true_density = stats.gaussian_kde(normal_samples)

    return origin, push, pushforward, bary_true_density


def lawgd_dk(mesh, n_mesh, dim, V, dV, ddV, epsilon, n_eigen=-1, domain=(-10, 10)):
    n = mesh.shape[0]
    if dim == 2:
        e = mesh[1,0]-mesh[0,0]
        ind = lambda x, epsilon=epsilon, n_mesh=n_mesh: \
            min((n_mesh*(n_mesh-int((x[1]-mesh[-1,-1])/e)-1) + int((x[0]-mesh[0,0])/e)), n-1)
    else:
        ind = lambda x, epsilon=epsilon, mesh=mesh, n_mesh=n_mesh: int((x-mesh[0])/epsilon)%n_mesh

    dV2 = np.sum(np.array([dV(m) for m in mesh])**2, axis=-1) if dim == 2 else dV(mesh)**2
    lapV = np.array([ddV(m) for m in mesh])

    if dim == 1:
        lap_eps = np.diag(-(4 if dim==2 else 2)*np.ones((n,)))+np.diag(np.ones((n-1,)), k=1)+np.diag(np.ones((n-1,)),k=-1)
        vs = np.diag(dV2 - 2*lapV)/4
        #if dim == 2:
        #    lap_eps = lap_eps + np.diag(np.ones((n-n_mesh,)), k=n_mesh) + np.diag(np.ones((n-n_mesh,)), k=-n_mesh)
        lap = -lap_eps / epsilon**2 + vs
    else:
        cte = -1/epsilon**2
        lap_eps = sparse.lil_matrix((n, n))
        lap_eps.setdiag(cte*np.ones((n,))*(-4), k=0)
        lap_eps.setdiag(cte*np.ones((n-1,)), k=1)
        lap_eps.setdiag(cte*np.ones((n-1,)), k=-1)
        lap_eps.setdiag(cte*np.ones((n-n_mesh,)), k=-n_mesh)
        lap_eps.setdiag(cte*np.ones((n-n_mesh,)), k=n_mesh)
        lap_eps.setdiag(lap_eps.diagonal(k=0) + (dV2-2*lapV)/4, k=0)
        lap_eps = lap_eps.tocsr()


    if dim == 1:
        eta_s, phi_s = eigh(lap)
        valid_indices = np.where(eta_s > 1e-2)[0]
        eta_s = eta_s[valid_indices]
        phi_s = phi_s.T[valid_indices]
        if n_eigen != -1:
            eta_s, phi_s = eta_s[:n_eigen], phi_s[:n_eigen]
        indic = -np.ones(mesh.shape)*(1e2)
        a, b = ind(domain[0]), ind(domain[1])
        indic[a:b] = 1
        rescale = np.repeat(np.exp(V(mesh)*indic/2).reshape((1,-1)), repeats=phi_s.shape[0], axis=0)
        rescale[:, :a] = 1
        rescale[:, b:] = 1
        phi = np.multiply(rescale, phi_s)
        eta = np.diag(1/eta_s)
        k = phi.T.dot(eta).dot(phi)
    else:
        if n_eigen <= 0:
            raise ArgumentError('In dim. 2, the number of eigenvalues must be > 0 and < n_mesh-1.')
        def in_domain(x): return domain[0] <= x[0] <= domain[1] and domain[0] <= x[1] <= domain[1]
        eta_s, phi_s = sparse.linalg.eigs(lap_eps, k=n_eigen, which='SM')
        phi_s = phi_s.T
        rescale = np.array([V(m) if in_domain(m) else 0 for m in mesh])
        for i in range(n_eigen):
            phi_s[i] = rescale*phi_s[i]
        phi_s = sparse.csr_matrix(phi_s)
        eta = sparse.csr_matrix(np.diag(1/eta_s))
        k = phi_s.T.dot(eta).dot(phi_s)


    if dim == 2:
        w = n_mesh
        dk = sparse.csr_matrix(k.shape)
        dk[0, :] = (k[1, :] - k[0, :], k[w-1, :] - k[0, :]).T / epsilon
        for i in range(1, n-w-1):
            dk[i, :] = (k[i+1,:] - k[i-1,:], k[i+w, :] - k[i, :]).T / (2*epsilon)
        dk[-1, :] = (k[-1,:] - k[-2,:], k[-1, :] - k[-1-w+1, :]).T / epsilon
    else:
        dk = np.zeros(k.shape)
        dk[0] = (k[1,:] - k[0,:])/epsilon
        for i in range(1, n-1):
            dk[i] = (k[i+1,:]-k[i-1,:])/(2*epsilon)
        dk[-1] = (k[-1,:]-k[-2,:])/epsilon

    dkf = lambda x, y: dk[ind(x), ind(y)]

    return dkf


def lawgd_gradient(dk, L, N, i):
    grad = np.zeros(L.shape[-1])
    for k in range(N):
        grad = grad + dk(L[i], L[k])
    return -grad / N


def laws_nnorms(means, stds, simul_params):
    pass


def gaussian_mixture_densities(means, sigmas):
    gmd = lambda x: np.array([np.exp(-0.5 * norm2(x - mean) / sigma**2) / (sigma * np.sqrt(2*np.pi)) for (mean, sigma) in zip(means, sigmas)]).reshape((-1, sigmas.shape[0]))
    return gmd


def gaussian_mixture(weights, means, sigmas):
    gm = lambda x: weights.dot(np.squeeze(gaussian_mixture_densities(means, sigmas)(x)))
    return gm


def gaussian_mixture_gradient(weights, means, sigmas, epsilon=1e-8):
    def mixture_gradient(x):
        dx = np.squeeze(gaussian_mixture_densities(means, sigmas)(x) + epsilon)
        x_factor = np.sum([w*d*(x - m) / s**2 for (m, s, w, d) in zip(means, sigmas, weights, dx)], axis=0)
        grad = x_factor / weights.dot(dx) # negative sign taken out because kernel.gradient expect V, not \log \pi = -V.

        return grad

    return mixture_gradient


def hue_lerp(t, rgb_start, rgb_end):
    hsv_start, hsv_end = color.rgb2hsv(rgb_start.astype(np.float64)), color.rgb2hsv(rgb_end.astype(np.float64))
    hue_start, hue_end = hsv_start[0], hsv_end[0]

    hue_delta = hue_end - hue_start
    if hue_start > hue_end:
        hue_start, hue_end = hue_end, hue_start
        t, hue_delta = 1-t, -hue_delta
    if hue_delta <= 0.5:
        hue_interp = hue_start + t*hue_delta
    else:
        hue_start += 1
        hue_interp = (hsv_start[0] + t*(hue_end - hue_start)) % 1

    saturation_interp = (1-t)*hsv_start[1] + t*hsv_end[1]
    value_interp = (1-t)*hsv_start[2] + t*hsv_end[1]

    hsv_interp = np.array([hue_interp, saturation_interp, value_interp])
    rgb_interp = color.hsv2rgb(hsv_interp)

    return rgb_interp


def mmd(a, b, x, y):
    kernel = lambda s, t: np.exp(-0.5*norm2(s - t)).item()
    num_samples_a, num_samples_x = a.shape[0], x.shape[0]

    if b.shape[0] != num_samples_a or y.shape[0] != num_samples_x:
        raise ValueError('a and b, and x and y must have the same size!')

    aa = np.sum(np.sum([kernel(a[i], b[j]) for i in range(num_samples_a) for j in range(num_samples_x)])) / num_samples_a**2
    ax = np.sum(np.sum([kernel(a[i], x[j]) for i in range(num_samples_a) for j in range(num_samples_x)])) / (num_samples_a*num_samples_x)
    xx = np.sum(np.sum([kernel(x[i], y[j]) for i in range(num_samples_x) for j in range(num_samples_x)])) / num_samples_x**2
    dist = np.sqrt(np.abs(aa + xx - 2*ax))

    return dist


if __name__ == '__main__':
    test_barycenter = False
    test_mixtures = False
    test_color = False
    test_c = False
    test_mmd = True

    if test_mmd:
        import matplotlib.pyplot as plt

        a = stats.norm.rvs(loc=0, scale=1, size=300)
        b = stats.norm.rvs(loc=0, scale=1, size=300)

        dists = []

        for t in range(10):
            x = stats.norm.rvs(loc=2/(t+1), scale=2/(2+1/(1+t)), size=300)
            y = stats.norm.rvs(loc=2/(t+1), scale=2/(2+1/(t+1)), size=300)
            dist = mmd(a, b, x, y)
            dists.append(dist)

        plt.grid()
        plt.plot(dists)
        plt.show()

    if test_c:
        lambdas = np.ones(2) * 0.5
        L = np.random.normal(loc=0, scale=1, size=300).reshape((2, 150, 1))
        intval = intc(L, lambdas)

        print('intval:', intval)


    if test_color:
        import matplotlib.pyplot as plt

        t = np.linspace(0, 1, 500)
        col_t = lambda t: hue_lerp(t, np.array([0, 1, 0]), np.array([1, 0, 0]))

        plt.grid()
        for i in range(len(t)-1):
            plt.plot([t[i], t[i+1]], [t[i], t[i+1]], c=col_t(t[i]))
        plt.show()


    if test_mixtures:
        import matplotlib.pyplot as plt

        weights = np.ones(2)*0.5
        means = np.array([-0, 3])
        sigmas = np.array([1, 0.5])
        mesh = np.linspace(-10, 10, 500)

        densities = lambda x: np.array([np.exp(-0.5 * norm2(x - mean) / sigma**2) / (sigma * np.sqrt(2*np.pi)) for (mean, sigma) in zip(means, sigmas)])
        mixture = np.vectorize(lambda x: weights.dot(densities(x)))
        factor = lambda x: weights * np.array([(x - mean) / sigma**2 for (mean, sigma) in zip(means, sigmas)])
        def mixture_gradient(x):
            dx = densities(x)
            grad = factor(x).dot(dx) / weights.dot(dx)

            return grad


    if test_barycenter:
        print('Testing barycenter_of_gaussians...')

        d = 32
        epsilon = 1e-1
        means = [np.zeros(d), np.zeros(d)]
        stds = [np.diag([1] + (d-1)*[epsilon]), np.diag([epsilon]*(d-1) + [1])]
        lambdas = [0.5, 0.5]

        mean, std, to = barycenter_of_gaussians(means, stds, lambdas)
        if d < 5:
            print('\tThresholded out:', to)
            print('\tBarycenter mean:\t', mean)
            print('\tBarycenter var:\n', std)
            print('\tBarycenter std:\n', np.sqrt(std))

        print('Testing barycenter_of_gaussians... [Ok]')
