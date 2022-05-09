import random
import math
import numpy as np
import numpy.random as rnd
import numpy.linalg as LA
from bpdca.bregman import BregmanProximalDifferenceOfConvex


def objective_function(x):
    return np.sum(np.power(np.power(np.dot(A, x), 2) - b, 2)) / 4 + np.sum(np.abs(x))


def gradient(x, y, gradient_kernel, l):
    Ay = np.dot(A, y)
    return -gradient_kernel(y) + ((Ay**3).dot(A) - np.dot(Ab, x))/l


def gradient_kernel(x):
    return np.sum(x**2) * x


def coefficient(v):
    dot = np.sum(v**2)
    return np.cbrt(1 / dot)


def kernel(x):
    return np.sum(x**2)**2 / 4


def setup(matrix_number, size):
    measurement = rnd.randn(matrix_number, size)

    sparse = math.ceil(size * .05)
    entry = rnd.randn(sparse)

    opt = np.zeros(size)
    for i, j in enumerate(random.sample(range(size), sparse)):
        opt[j] = entry[i]
    opt = opt / LA.norm(opt)
    amplitude = np.power(np.dot(measurement, opt), 2)

    return measurement, amplitude, opt


def l_smooth(matrix, vec, setting=""):
    num, size = matrix.shape
    if setting == "bpdc":
        return 3 * sum([np.dot(matrix[i], matrix[i]) ** 2 for i in range(num)])
    elif setting == "bpg":
        return 3 * sum([np.dot(matrix[i], matrix[i]) ** 2 for i in range(num)]) \
               + sum([np.dot(matrix[i], matrix[i]) * abs(vec[i]) for i in range(num)])
    elif setting == "bpdc_sum":
        return 3 * LA.norm(sum([np.dot(matrix[i], matrix[i]) * np.outer(matrix[i], matrix[i]) for i in range(num)]), 2)
    elif setting == "bpdc_fast":
        return 9 * LA.norm(matrix.T.dot(matrix), 2)


def power_method(A, AT, b, d):
    v = rnd.randn(d)
    v = v / LA.norm(v)
    for _ in range(50):
        u = AT(b * A(v))
        v = u / LA.norm(u)
    return v


if __name__ == '__main__':
    m = 10000
    d = 20

    A, b, x_opt = setup(m, d)
    Ab = np.dot(A.T, np.diag(b)).dot(A)
    x_0 = np.abs(power_method(A.dot, A.T.dot, b, d))
    L_bpdc = l_smooth(A, b, "bpdc_fast")
    bdc = BregmanProximalDifferenceOfConvex(x_0, L_bpdc, gradient, coefficient, kernel, gradient_kernel, with_regularizer="soft", with_normalization=True)
    bdc.dc_algorithm(with_extrapolation=True)
    print("||x - opt|| / ||opt|| = ", LA.norm(x_opt - bdc.x_k) / LA.norm(x_opt))
    print(objective_function(bdc.x_k), objective_function(x_opt))
