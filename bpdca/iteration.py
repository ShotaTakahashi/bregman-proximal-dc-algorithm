import numpy as np
import numpy.linalg as LA
from abc import ABCMeta


class Iteration(metaclass=ABCMeta):
    def __init__(self, x_0):
        self.x_k, self.x_k_old = x_0, x_0
        self.iter = 0


class IterativeMethod(Iteration):
    def __init__(self, x_0):
        Iteration.__init__(self, x_0)
        self.TOL = 1e-6
        self.MAX_ITER = 9000

    def stop(self):
        return LA.norm(self.x_k - self.x_k_old) / max(1.0, LA.norm(self.x_k)) < self.TOL


class Extrapolation(Iteration):
    def __init__(self, x_0, restart_iter=200):
        Iteration.__init__(self, x_0)
        self.y_k = x_0
        self.beta_k, self.__theta, self.__theta_old = 0.0, 1.0, 1.0
        self.restart_iter = restart_iter
        self.__restart = self.restart_iter

    def adaptive_scheme(self):
        return np.dot(self.y_k - self.x_k, self.x_k - self.x_k_old) > 0

    def beta_update(self):
        self.beta_k = (self.__theta_old - 1) / self.__theta
        self.y_k = self.x_k + self.beta_k * (self.x_k - self.x_k_old)
        self.__theta_update()
        self.__theta_restart()

    def __theta_update(self):
        self.__theta_old, self.__theta = self.__theta, (1 + (1 + 4 * self.__theta**2)**0.5) * 0.5

    def __theta_restart(self):
        if self.adaptive_scheme():
            self.beta_k, self.__theta, self.__theta_old = 0.0, 1.0, 1.0
        if self.iter == self.restart_iter:
            self.beta_k, self.__theta, self.__theta_old = 0.0, 1.0, 1.0
            self.restart_iter += self.__restart
