import numpy as np
from commons import epsilon


class Model:
    coef_full_ = None
    coef_ = None
    params_ = None

    def __init__(self, coef, from_matlab=True, params={}):
        if from_matlab:
            self.coef_full_ = np.array(coef._data).reshape(1, coef.size[0])[0]
        else:
            self.coef_full_ = coef
        self.coef_ = self.coef_full_
        self.coef_[abs(self.coef_) < epsilon] = 0.0
        self.params_ = params

    def predict(self, X):
        return np.sum(X * self.coef_, axis=1)
