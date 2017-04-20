from commons import tlp_n_folds as n_folds, tlp_n_deltas1 as n_deltas1, tlp_n_deltas2 as n_deltas2, tlp_n_taus as n_taus
from models import Model
import matlab.engine
import numpy as np


def fit_ttlp(setup, matlab_engine, lasso_fit):
    t = max(abs(lasso_fit.coef_))
    p = setup.x_tune.shape[1]
    g = len(setup.network)
    deltas1 = np.linspace(start=t, stop=p * t / 4, num=n_deltas1)
    deltas2 = np.linspace(start=t, stop=t * g, num=n_deltas2)
    taus = np.linspace(start=1e-6, stop=t / 2, num=n_taus)

    m_wt = matlab.double(np.sqrt(setup.degrees).tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_b0 = matlab.double(lasso_fit.coef_.tolist(), size=(p, 1))
    m_del1 = matlab.double(deltas1.tolist())
    m_del2 = matlab.double(deltas2.tolist())
    m_taus = matlab.double(taus.tolist())

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())
    del1, del2, tau = matlab_engine.cvTlp(m_y, m_X, m_wt, m_netwk, m_b0, m_del1, m_del2, m_taus, 0, n_folds, nargout=3)

    # Training
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.tlp(m_y, m_X, m_wt, m_netwk, m_b0, del1, del2, tau, tau)

    return Model(coef, params={"delta 1": del1, "delta 2": del2, "tau": tau})


def fit_ltlp(setup, matlab_engine, lasso_fit):
    t = max(abs(lasso_fit.coef_))
    p = setup.x_tune.shape[1]
    g = len(setup.network)
    deltas1 = np.linspace(start=t, stop=p * t / 4, num=n_deltas1)
    deltas2 = np.linspace(start=t, stop=t * g, num=n_deltas2)
    taus = np.linspace(start=1e-6, stop=t / 2, num=n_taus)

    m_wt = matlab.double(np.sqrt(setup.degrees).tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_b0 = matlab.double(lasso_fit.coef_.tolist(), size=(p, 1))
    m_del1 = matlab.double(deltas1.tolist())
    m_del2 = matlab.double(deltas2.tolist())
    m_taus = matlab.double(taus.tolist())

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())
    del1, del2, tau = matlab_engine.cvTlp(m_y, m_X, m_wt, m_netwk, m_b0, m_del1, m_del2, m_taus, 1, n_folds, nargout=3)

    # Training
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.tlp(m_y, m_X, m_wt, m_netwk, m_b0, del1, del2, 100, tau)

    return Model(coef, params={"delta 1": del1, "delta 2": del2, "tau": tau})