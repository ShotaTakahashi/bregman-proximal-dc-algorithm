import csv
import numpy.linalg as LA
from abc import ABCMeta, abstractmethod
from bpdca.iteration import IterativeMethod, Extrapolation

LEAST_ITER = 0


class DifferenceOfConvex(IterativeMethod, metaclass=ABCMeta):
    def __init__(self, x_0, update):
        IterativeMethod.__init__(self, x_0)
        self.update = update

    @abstractmethod
    def dc_algorithm(self):
        pass


class DifferenceOfConvexExtrapolation(DifferenceOfConvex, Extrapolation):
    def __init__(self, x_0, update):
        DifferenceOfConvex.__init__(self, x_0, update)
        Extrapolation.__init__(self, x_0)

    def dc_algorithm(self, with_extrapolation=True, output=False, path="", opt=""):
        if output:
            f = open(path, "a")
            csvWriter = csv.writer(f, lineterminator='\n')

        for _ in range(self.MAX_ITER):
            self.iter += 1

            if with_extrapolation:
                self.beta_update()
            else:
                self.beta_k = 0.0

            self.y_k = self.x_k + self.beta_k * (self.x_k - self.x_k_old)
            self.x_k_old, self.x_k = self.x_k, self.update(self.x_k, self.y_k)

            if output:
                csvWriter.writerow(["bpdcae", self.x_k.size, self.iter,
                                    LA.norm(self.x_k - opt) / max(1.0, LA.norm(opt))])
            if self.stop() and self.iter > LEAST_ITER:
                return self.x_k
        if output:
            f.close()
        return self.x_k
