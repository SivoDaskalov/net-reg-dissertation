from commons import cv_n_folds
from models import Model
import matlab.engine
import numpy as np

n_deltas1 = 5
n_deltas2 = 5
n_thetas = 6


def fit_tlpi(setup, engine, lasso_fit):
    t = max(abs(lasso_fit.coef_))
    p = setup.x_tune.shape[1]
    g = len(setup.network)
    deltas1 = np.linspace(start=t, stop=p * t / 4, num=n_deltas1)
    deltas2 = np.linspace(start=t, stop=t * g, num=n_deltas2)
    thetas = np.linspace(start=1e-6, stop=t / 2, num=n_thetas)

    m_wt = matlab.double(np.sqrt(setup.degrees).tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])

    return None

# def fit_grace(setup, matlab_engine):
    m_adj = matlab.double([1] * len(setup.network))
    m_wt = matlab.double(np.sqrt(setup.degrees).tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_lam1 = matlab.double(lambdas1)
    m_lam2 = matlab.double(lambdas2)

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())
    cv_lam1, cv_lam2 = matlab_engine.cvGrace(m_y, m_X, m_wt, m_netwk, m_adj, m_lam1, m_lam2, n_folds, nargout=2)

    # Training
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.grace(m_y, m_X, m_wt, m_netwk, m_adj, cv_lam1, cv_lam2)

    return Model(coef, params={"lambda 1": cv_lam1, "lambda 2": cv_lam2})
