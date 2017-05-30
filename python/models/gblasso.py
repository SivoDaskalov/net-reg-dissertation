from commons import cv_n_folds as n_folds, gblasso_lambda_values as lambdas, gblasso_gamma_values as gammas, \
    gblasso_gamma_opt as opt_gamma, gblasso_lambda_opt as opt_lambda, gblasso_train_maxiter, gblasso_tune_maxiter, \
    timestamp, gblasso_real_maxiter
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from models import Model
import numpy as np
from scipy.optimize import minimize

minimization_method = "BFGS"


def fit_gblasso(setup):
    wt = np.sqrt(setup.degrees)
    # Tuning
    lam, gam = cvGblasso(setup.y_tune, setup.x_tune, wt, setup.network, lambdas, gammas)

    # Training
    coef = gblasso(setup.y_train, setup.x_train, wt, setup.network, lam, gam, gblasso_train_maxiter)

    return Model(coef, params={"lambda": lam, "gamma": gam}, from_matlab=False)


def cvGblasso(Y, X, wt, network, lambdas, gammas):
    best_mse = 999999
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    for lam in lambdas:
        for gam in gammas:
            errors = []
            for training, holdout in kf.split(X):
                coef = gblasso(Y[training], X[training, :], wt, network, lam, gam, gblasso_tune_maxiter)
                errors.append(mean_squared_error(Y[holdout], np.sum(X[holdout] * coef, axis=1)))
            mse = np.mean(errors)
            print("%sLambda = %.2f,\t Gamma = %.2f,\t MSE = %.2f" % (timestamp(), lam, gam, mse))
            if mse < best_mse:
                best_mse = mse
                best_lam = lam
                best_gam = gam
    return best_lam, best_gam


def gblasso(Y, X, wt, network, lam, gam, maxiter = 0):
    b0 = np.zeros(X.shape[1])
    if maxiter == 0:
        return minimize(gblasso_penalty, b0, (Y, X, wt, network, lam, gam), method=minimization_method).x
    return minimize(gblasso_penalty, b0, (Y, X, wt, network, lam, gam), method=minimization_method,
                    options={"maxiter": maxiter}).x


def gblasso_penalty(b, Y, X, wt, network, lam, gam):
    errors = sum(np.power(np.sum(X * b, axis=1) - Y, 2))
    network_penalty = sum(np.power(
        [(abs(b[i1 - 1]) / wt[i1 - 1]) ** gam + (abs(b[i2 - 1]) / wt[i2 - 1]) ** gam for (i1, i2) in network],
        1.0 / gam))
    return errors + lam * network_penalty


def fit_gblasso_opt(setup):
    return param_fit_gblasso(setup, opt_lambda, opt_gamma, maxiter=gblasso_real_maxiter)


def param_fit_gblasso(setup, lam, gam, use_tuning_set=False, maxiter=None):
    if maxiter is None:
        maxiter = gblasso_train_maxiter
    wt = np.sqrt(setup.degrees)
    if use_tuning_set:
        coef = gblasso(setup.y_tune, setup.x_tune, wt, setup.network, lam, gam, maxiter)
    else:
        coef = gblasso(setup.y_train, setup.x_train, wt, setup.network, lam, gam, maxiter)
    return Model(coef, params={"lambda": lam, "gamma": gam}, from_matlab=False)
