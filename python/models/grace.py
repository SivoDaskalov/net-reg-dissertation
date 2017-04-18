from sklearn.linear_model import LinearRegression
from commons import cv_nfolds
from commons import epsilon
from models import Model
import matlab.engine
import numpy as np

lam1_values = [10 ** x for x in range(-2, 5)]
lam2_values = [10 ** x for x in range(-2, 5)]


def fit_grace(setup, matlab_engine):
    m_adj = matlab.double([1] * len(setup.network))
    m_wt = matlab.double(np.sqrt(setup.degrees).tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_lam1 = matlab.double(lam1_values)
    m_lam2 = matlab.double(lam2_values)

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())
    lam1, lam2 = matlab_engine.cvGrace(m_y, m_X, m_wt, m_netwk, m_adj, m_lam1, m_lam2, cv_nfolds, nargout=2)

    # Training
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.grace(m_y, m_X, m_wt, m_netwk, m_adj, lam1, lam2)

    return Model(coef, params={"lam1": lam1, "lam2": lam2})


def fit_agrace(setup, matlab_engine, enet_fit=None):
    n = setup.x_tune.shape[0]
    p = setup.x_tune.shape[1]
    if p < n:
        b0 = LinearRegression(fit_intercept=False).fit(X=setup.x_tune, y=setup.y_tune).coef_
    else:
        b0 = enet_fit.coef_
    adj = [1.0 if b0[p1 - 1] * b0[p2 - 1] > 0 and b0[p1 - 1] > epsilon and b0[p2 - 1] > epsilon else -1.0
           for (p1, p2) in setup.network]

    m_adj = matlab.double(adj)
    m_wt = matlab.double(np.sqrt(setup.degrees).tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_lam1 = matlab.double(lam1_values)
    m_lam2 = matlab.double(lam2_values)

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())
    lam1, lam2 = matlab_engine.cvGrace(m_y, m_X, m_wt, m_netwk, m_adj, m_lam1, m_lam2, cv_nfolds, nargout=2)

    # Training
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.grace(m_y, m_X, m_wt, m_netwk, m_adj, lam1, lam2)

    return Model(coef, params={"lam1": lam1, "lam2": lam2})
