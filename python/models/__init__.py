import numpy as np


class Model:
    coef_ = None
    params_ = None

    def __init__(self, coef, from_matlab=True, params={}):
        if from_matlab:
            self.coef_ = np.array(coef._data).reshape(1, coef.size[0])[0]
        else:
            self.coef_ = coef
        self.params_ = params

    def predict(self, X):
        return np.sum(X * self.coef_, axis=1)
