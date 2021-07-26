import numpy.linalg as LA
from bpdca.iteration import Extrapolation
from bpdca.dca import DifferenceOfConvexExtrapolation
from bpdca.toolbox.tool import *

LEAST_ITER = 0


class Bregman:
    def __init__(self, kernel, grad):
        self.kernel = kernel
        self.grad_kernel = grad
        self.bregman_distance = bregman_distance(kernel, grad)


class BregmanExtrapolation(Bregman, Extrapolation):
    def __init__(self, x_0, kernel, gradient_kernel, restart_iter=200):
        Bregman.__init__(self, kernel, gradient_kernel)
        Extrapolation.__init__(self, x_0, restart_iter)
        self.rho = .99

    def adaptive_scheme(self):
        return self.bregman_distance(self.x_k, self.y_k) \
               > self.rho * self.bregman_distance(self.x_k_old, self.x_k)


class BregmanProximalDifferenceOfConvex(DifferenceOfConvexExtrapolation, BregmanExtrapolation):
    def __init__(self, x_0, l_smad, grad, coefficient, kernel, grad_kernel, restart_iter=200,
                 with_normalization=False,
                 with_regularizer="soft",
                 reg_para=1.0,
                 ):
        DifferenceOfConvexExtrapolation.__init__(self, x_0, self.update)
        BregmanExtrapolation.__init__(self, x_0, kernel, grad_kernel, restart_iter)
        self.l_smad = l_smad
        self.grad = grad
        self.coefficient = coefficient
        self.reg_para = reg_para
        self.with_regularizer = with_regularizer
        self.with_normalization = with_normalization

    def update(self, x, y):
        p = self.grad(x, y, self.grad_kernel, self.l_smad)
        if self.with_regularizer == "soft":
            v = soft_thresholding(p, self.reg_para / self.l_smad)
        else:
            v = p
        t = self.coefficient(v)
        z = -t * v
        if self.with_normalization:
            z = z / LA.norm(z)
        return z

