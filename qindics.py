import pdb
import numpy as np
import matplotlib.pyplot as plt


# Assumes that q is negative on the interior of the region.
def smooth_indicator(q, dim, epsilon=1):
    if dim == 1:
        phi = lambda u: np.exp(-q(u) / epsilon)
    else:
        phi = lambda u, v: np.exp(-q(u, v) / epsilon)

    return phi


def conique(x, y, q, ell, cte):
    return q[0, 0]*x**2 + 2*q[1, 0]*x*y + q[1, 1]*y**2 + ell[0]*x + ell[1]*y + cte


def square(x, y):
    return (y**2 -2*(1 + 0.5*x**2 - 0.33*x**4))*(x**2 - 2*(1 + 0.5*y**2 - 0.33*y**4)) - 100


def two_disks_2d(x, y, a, b, c, d):
    return ((x-a)**2 + (y-b)**2 - 1)*((x-c)**2 + (y-d)**2 - 1)


def unnormdensity_2disks_2d(a, b, c, d):
    theta = lambda x: np.exp(-10 / (2*x+1)**8)
    set_eq = lambda x, y: 100*theta(two_disks_2d(x, y, a, b, c, d) / 50)
    indic = lambda u, v: np.exp(-set_eq(u, v))

    return indic


def potential_2disks_2d(a, b, c, d):
    theta = lambda x: np.exp(-10 / (2*x+1)**8)
    set_eq = lambda x, y: 100*theta(two_disks_2d(x, y, a, b, c, d) / 50)

    return set_eq


def potential_gradient_2disks_2d(a, b, c, d):
    theta = lambda x: np.exp(-10 / (2*x+1)**8)
    f = lambda x, y: two_disks_2d(x, y, a, b, c, d)
    pot = lambda x, y: 320*(theta(f(x, y) / 50) / (f(x, y) / 25 + 1)**9) * 2 * np.array([
        (x - c)*((x - a)**2 + (y - b)**2 - 1) + (x - a)*((x - c)**2 + (y - d)**2 - 1),
        (y - d)*((x - a)**2 + (y - b)**2 - 1) + (y - b)*((x - c)**2 + (y - d)**2 - 1)
    ])

    return pot


if __name__ == '__main__':
    x_shifts, y_shifts = [2, -2, 2, -2], [2, -2, -2, 2]
    indics = []

    x0, y0 = np.linspace(-5, 5, 200), np.linspace(-5, 5, 200)
    x, y = np.meshgrid(x0, y0)
    riem = ((5 - (-5)) / x0.size)**2

    for i in range(2):
        indic_func = unnormdensity_2disks_2d(x_shifts[i], y_shifts[i], x_shifts[i+2], y_shifts[i+2])

        indic = indic_func(x, y)
        indic = indic / np.sum(np.sum(indic)*riem)
        indics.append(indic)

    mode = '3d'
    if mode == '3d':
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    else:
        fig, ax = plt.subplots()
    levels = np.linspace(indic.min(), indic.max(), 10)
    final = indics[0] + indics[1]
    if mode == '3d':
        m = ax.plot_surface(x, y, final)
    else:
        m = ax.contourf(x, y, final, levels=levels)
    ax.grid()
    plt.colorbar(mappable=m, ax=ax)

    plt.show()
