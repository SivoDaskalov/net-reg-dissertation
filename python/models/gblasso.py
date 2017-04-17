from commons import cross_validation_folds
from models import Model
import numpy as np
import matlab.engine
from scipy.optimize import minimize

lam_values = [10 ** x for x in range(-2, 3)]
gam_values = [2 * x for x in range(1, 6)]

import math


def generate_true_coefficients(trans_factor_coefficients, regulated_denominator, n_regulated_negative,
                               n_regulated_positive, n_trailing_zero_genes):
    leading_coefficients = [[tf] + [-tf / regulated_denominator] * n_regulated_negative +
                            [tf / regulated_denominator] * n_regulated_positive for tf in trans_factor_coefficients]
    true_coefficients = [coef for sublist in leading_coefficients for coef in sublist] + [0] * n_trailing_zero_genes
    return true_coefficients


def fit_gblasso(setup, matlab_engine):
    true_coef = generate_true_coefficients([5, -5, 3, -3], math.sqrt(10), 0, 10, 176)
    estimated_coef = minimize(gblasso_penalty, np.zeros(setup.x_tune.shape[1]),
                    (setup.y_tune, setup.x_tune, setup.degrees, setup.network, 5, 2), method='BFGS').x
    print(estimated_coef)
    # m_wt = matlab.double(setup.degrees.tolist(), size=(setup.x_train.shape[1], 1))
    # m_netwk = matlab.double([[p1, p2] for (p1, p2) in setup.network])
    # m_lam = matlab.double(lam_values)
    # m_gam = matlab.double(gam_values)
    #
    # # Tuning
    # m_y = matlab.double(setup.y_tune.tolist(), size=(len(setup.y_tune), 1))
    # m_X = matlab.double(setup.x_tune.tolist())
    #
    # matlab_engine.workspace['Y'] = m_y
    # matlab_engine.workspace['X'] = m_X
    # matlab_engine.workspace['wt'] = m_wt
    # matlab_engine.workspace['netwk'] = m_netwk
    # matlab_engine.workspace['lam'] = 2.0
    # matlab_engine.workspace['gam'] = 2.0
    #
    # coef, lam, gam, mse = matlab_engine.cvGblasso(m_y, m_X, m_wt, m_netwk, m_lam, m_gam,
    #                                               float(cross_validation_folds), nargout=4)
    #
    # # Training
    # m_y = matlab.double(setup.y_train.tolist(), size=(len(setup.y_train), 1))
    # m_X = matlab.double(setup.x_train.tolist())
    # coef = matlab_engine.gblasso(m_y, m_X, m_wt, m_netwk, lam, gam)
    #
    # return Model(coef, params={"lam": lam, "gam": gam})


def gblasso_penalty(b, Y, X, wt, network, lam, gam):
    sum_square_errors = sum(np.power(np.sum(X * b, axis=1) - Y, 2))
    network_penalty_terms = [abs(b[i1 - 1]) ** gam / wt[i1 - 1] + abs(b[i2 - 1]) ** gam / wt[i2 - 1]
                             for (i1, i2) in network]
    network_penalty = lam * 2.0 ** (1.0 - (1.0 / gam)) * sum(network_penalty_terms) ** (1.0 / gam)
    return sum_square_errors + network_penalty
