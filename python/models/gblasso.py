from commons import cross_validation_folds
from models import Model
import matlab.engine

lam_values = [10 ** x for x in range(-2, 3)]
gam_values = [2 * x for x in range(1, 6)]

def fit_gblasso(setup, matlab_engine):
    m_wt = matlab.double(setup.degrees.tolist(), size=(setup.x_train.shape[1], 1))
    m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    m_lam = matlab.double(lam_values)
    m_gam = matlab.double(gam_values)

    # Tuning
    m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    m_X = matlab.double(setup.x_tune.tolist())

    matlab_engine.workspace['Y'] = m_y
    matlab_engine.workspace['X'] = m_X
    matlab_engine.workspace['wt'] = m_wt
    matlab_engine.workspace['netwk'] = m_netwk
    matlab_engine.workspace['lam'] = 2.0
    matlab_engine.workspace['gam'] = 2.0

    coef, lam, gam, mse = matlab_engine.cvGblasso(m_y, m_X, m_wt, m_netwk, m_lam, m_gam,
                                                  float(cross_validation_folds), nargout=4)

    # Training
    m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    m_X = matlab.double(setup.x_train.tolist())
    coef = matlab_engine.gblasso(m_y, m_X, m_wt, m_netwk, lam, gam)

    return Model(coef, params={"lam": lam, "gam": gam})
