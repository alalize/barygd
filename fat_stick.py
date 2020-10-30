import pdb
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from utils import norm, norm2


def fat_stick(x, y, a, b, n, eta=1e1):
    s = 0
    sg_k = np.eye(2) * np.sqrt((n-1) / norm(b-a))
    for k in range(n):
        mu_k = a + k*(b-a)/(n-1)
        exp_k = np.exp(-eta*0.5*( sg_k[0,0]*(x-mu_k[0])**2 + 2*sg_k[0,1]*(x-mu_k[0])*(y-mu_k[1]) \
            + sg_k[1,1]*(y-mu_k[1])**2 ))
        s = s + exp_k
    return s


def idfat_stick(x, a, b, n, eta=1e1):
    grad = np.zeros(shape=x.shape)
    sg_k = np.eye(2) * np.sqrt((n-1)/norm(b-a))
    denom = 0

    for k in range(n):
        mu_k = a + k*(b-a)/(n-1)
        exp_k = np.exp(-0.5*eta*(x-mu_k).dot(sg_k).dot(x-mu_k))
        denom = denom + exp_k
    for k in range(n):
        mu_k = a + k*(b-a)/(n-1)
        exp_k = eta*np.exp(-eta*0.5*(x-mu_k).dot(sg_k).dot(x-mu_k))*sg_k.dot(x-mu_k)
        grad = grad + exp_k
    grad = grad / denom

    return grad


def dfat_stick(a, b, n, eta=1e1):
    return lambda x,a=a,b=b,n=n,eta=eta: idfat_stick(x, a, b, n, eta)

"""
def fat_stick(a, b, eta, power=2):
    stick = lambda x, y: np.exp(-(0.5/eta)*( ( (x-a[0])**2 + (y-a[1])**2 )**(power/2) / \
        ( 1 + ( ( (x-a[0])*(b[0]-a[0]) + (y-a[1])*(b[1]-a[1]) ) / \
        ( np.sqrt((x-a[0])**2 + (y-a[1])**2) * np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2) ) )**2 )**power))

    return stick


def dfat_stick(a, b, eta):
    term = lambda x: (x-a).dot(b-a) / (norm(x-a)*norm(b-a))
    denom = lambda x: 1 + term(x)**2
    frac = lambda x: term(x)**2 / (1 + term(x)**2)
    grad = lambda x: (1/eta)*( (x-a)*(0.5 + frac(x)) -2*frac(x)*norm2(x-a)*(b-a) )

    return grad
"""


if __name__ == '__main__':
    x = np.linspace(-6, 6, 300)
    y = np.linspace(-6, 6, 300)
    X, Y = np.meshgrid(x, y)

    a, A = np.array([0, 0]), np.array([2, 1])
    b, B = np.array([1, 1]), np.array([3, 0])

    eta = 20
    stick = fat_stick(X, Y, a, b, 100)
    stick2 = fat_stick(X, Y, A, B, 100)

    plt.contourf(X, Y, stick, cmap='hot', levels=20, alpha=0.5)
    plt.contourf(X, Y, stick2, cmap='hot', levels=20, alpha=0.5)

    plt.show()

    """
    Eta = [1, 1]
    power = 2

    plt.grid()
    for k in range(2):
        plt.contourf(X, Y, fat_stick(A[k], B[k], Eta[k])(X, Y), cmap='hot', levels=50, alpha=0.5)
    plt.show()
    """
