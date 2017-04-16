import numpy as np


class Model:
    coef_ = None

    def __init__(self, coef, column_coef = True, params = {}):
        self.coef_ = coef
        if column_coef:
            self.coef_.reshape((1, self.coef_.size[0]))
        self.params_ = params

    def predict(self, X):
        return np.sum(X * self.coef_, axis=1)