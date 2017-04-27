from commons import cv_n_folds as n_folds, epsilon, grace_lambda1_values as lambdas1, grace_lambda2_values as lambdas2, \
    grace_lambda1_opt as opt_lambda1, grace_lambda2_opt as opt_lambda2
from sklearn.linear_model import LinearRegression
from models import Model
import matlab.engine
import numpy as np


def fit_grace(setup, matlab_engine):
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


def fit_grace_opt(setup, matlab_engine):
    m_adj = matlab.double([1] * len(setup.network))
    m_wt = matlab.double(np.sqrt(setup.degrees).tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.grace(m_y, m_X, m_wt, m_netwk, m_adj, opt_lambda1, opt_lambda2)
    return Model(coef, params={"lambda 1": opt_lambda1, "lambda 2": opt_lambda2})


def fit_agrace_opt(setup, matlab_engine, enet_fit=None):
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
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.grace(m_y, m_X, m_wt, m_netwk, m_adj, opt_lambda1, opt_lambda2)
    return Model(coef, params={"lambda 1": opt_lambda1, "lambda 2": opt_lambda2})
