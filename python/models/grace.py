from commons import cross_validation_folds
from models import Model
from sklearn.linear_model import LinearRegression
import matlab.engine
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy.optimize import minimize

lam1_values = [10 ** x for x in range(-2, 5)]
lam2_values = [10 ** x for x in range(-2, 5)]
minimize_method = "BFGS"


def fit_grace(setup, matlab_engine):
    m_adj = matlab.double([1] * len(setup.network))
    m_wt = matlab.double(setup.degrees.tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_lam1 = matlab.double(lam1_values)
    m_lam2 = matlab.double(lam2_values)

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())
    lam1, lam2 = matlab_engine.cvGrace(m_y, m_X, m_wt, m_netwk, m_adj, m_lam1, m_lam2,
                                                  float(cross_validation_folds), nargout=2)

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
    adj = [1.0 if b0[p1-1] * b0[p2-1] > 0 else -1.0 for (p1, p2) in setup.network]

    m_adj = matlab.double(adj)
    m_wt = matlab.double(setup.degrees.tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_lam1 = matlab.double(lam1_values)
    m_lam2 = matlab.double(lam2_values)

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())
    lam1, lam2 = matlab_engine.cvGrace(m_y, m_X, m_wt, m_netwk, m_adj, m_lam1, m_lam2,
                                                  float(cross_validation_folds), nargout=2)

    # Training
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.grace(m_y, m_X, m_wt, m_netwk, m_adj, lam1, lam2)

    return Model(coef, params={"lam1": lam1, "lam2": lam2})


def fit_py_grace(setup):
    adj = [1] * len(setup.network)
    wt = np.sqrt(setup.degrees)

    # Tuning
    lam1, lam2 = cvGrace(setup.y_tune, setup.x_tune, wt, setup.network, adj, lam1_values, lam2_values)

    # Training
    coef = grace(setup.y_train, setup.x_train, wt, setup.network, adj, lam1, lam2)

    return Model(coef, params={"lam1": lam1, "lam2": lam2}, from_matlab=False)


def cvGrace(Y, X, wt, network, a, lam1_values, lam2_values):
    best_mse = 999999
    for lam1 in lam1_values:
        for lam2 in lam2_values:
            kf = KFold(n_splits=cross_validation_folds, shuffle=True, random_state=1)
            errors = []
            for training, holdout in kf.split(X):
                coef = grace(Y[training], X[training,:], wt, network, a, lam1, lam2)
                errors.append(mean_squared_error(Y[holdout], np.sum(X[holdout] * coef, axis=1)))
            mse = np.mean(errors)
            if mse < best_mse:
                best_mse = mse
                best_lam1 = lam1
                best_lam2 = lam2
    return best_lam1, best_lam2


def grace(Y, X, wt, network, a, lam1, lam2):
    b0 = np.zeros(X.shape[1])
    return minimize(grace_penalty, b0, (Y, X, wt, network, a, lam1, lam2), method=minimize_method).x


def grace_penalty(b, Y, X, wt, network, a, lam1, lam2):
    errors = sum(np.power(np.sum(X * b, axis=1) - Y, 2))
    l1_norm = lam1 * sum(abs(b))
    grace_pen = lam2 * sum([(b[network[e][0]] / wt[network[e][0]] + a[e] * b[network[e][1]] / wt[network[e][1]]) ** 2
                            for e in range(len(network))])
    return errors + l1_norm + grace_pen
