from commons import cv_n_folds
from models import Model
import matlab.engine
import numpy as np


def fit_tlpi(setup, engine):
    return None


# def fit_grace(setup, matlab_engine):
#     m_adj = matlab.double([1] * len(setup.network))
#     m_wt = matlab.double(np.sqrt(setup.degrees).tolist(), size=(setup.x_train.shape[1], 1))
#     m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
#     m_lam1 = matlab.double(lambdas1)
#     m_lam2 = matlab.double(lambdas2)
#
#     # Tuning
#     m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
#     m_X = matlab.double(setup.x_tune.tolist())
#     cv_lam1, cv_lam2 = matlab_engine.cvGrace(m_y, m_X, m_wt, m_netwk, m_adj, m_lam1, m_lam2, n_folds, nargout=2)
#
#     # Training
#     m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
#     m_X = matlab.double(setup.x_train.tolist())
#     coef = matlab_engine.grace(m_y, m_X, m_wt, m_netwk, m_adj, cv_lam1, cv_lam2)
#
#     return Model(coef, params={"lambda 1": cv_lam1, "lambda 2": cv_lam2})
