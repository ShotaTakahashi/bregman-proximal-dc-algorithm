import numpy as np
import numpy.linalg as LA


def soft_thresholding(x, tau):
    return np.maximum(np.abs(x) - tau, 0.0)*np.sign(x)


def cubic_eq(p, q):
    disc = np.sqrt(q**2/4 + p**3/27)
    ans = np.cbrt(-q/2+disc) + np.cbrt(-q/2 - disc)
    return ans


def cubic_eq_without_normalization(u):
    norm = LA.norm(u)**2
    if norm == 0:
        return 1
    return cubic_eq(1/norm, -1/norm)


def cubic_eq_with_normalization(u):
    norm = LA.norm(u)**2
    if norm == 0:
        return 1
    return cubic_eq(2/norm, -1/norm)


def bregman_distance(kernel, grad):
    return lambda x, y: kernel(x) - kernel(y) - np.dot(grad(y), x - y)


def bregman_distance_complex(kernel, grad):
    return lambda x, y: kernel(x) - kernel(y) - 2*np.real(np.dot(grad(y), np.conj(x - y)))
