from commons import cross_validation_folds
from models import Model
import matlab.engine

lam1_values = [10**x for x in range(-2,3)]
lam2_values = [10**x for x in range(-2,3)]


def fit_grace(setup, matlab_engine):
    m_adj = matlab.double([1] * len(setup.network))
    m_wt = matlab.double(setup.degrees.tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_lam1 = matlab.double(lam1_values)
    m_lam2 = matlab.double(lam2_values)

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())
    coef, lam1, lam2, mse = matlab_engine.cvGrace(m_y, m_X, m_wt, m_netwk, m_adj, m_lam1, m_lam2,
                                                  float(cross_validation_folds), nargout=4)

    # Training
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.grace(m_y, m_X, m_wt, m_netwk, m_adj, lam1, lam2)

    return Model(coef, params={"lam1": lam1, "lam2": lam2})

def fit_agrace(setup, matlab_engine):
    m_adj = matlab.double([1] * len(setup.network))
    m_wt = matlab.double(setup.degrees.tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_lam1 = matlab.double(lam1_values)
    m_lam2 = matlab.double(lam2_values)

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())
    coef, lam1, lam2, mse = matlab_engine.cvGrace(m_y, m_X, m_wt, m_netwk, m_adj, m_lam1, m_lam2,
                                                  float(cross_validation_folds), nargout=4)

    # Training
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.grace(m_y, m_X, m_wt, m_netwk, m_adj, lam1, lam2)

    return Model(coef, params={"lam1": lam1, "lam2": lam2})