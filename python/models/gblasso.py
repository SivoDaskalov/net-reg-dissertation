from commons import cv_nfolds
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from models import Model
import numpy as np
from scipy.optimize import minimize

lam_values = [10 ** x for x in range(-2, 5)]
gam_values = [2.0, 3.0]
folds = 5 # Reduced number of folds due to the heavy calculations
minimize_method = "BFGS"


def fit_gblasso(setup):
    # Tuning
    lam, gam = cvGblasso(setup.y_tune, setup.x_tune, setup.degrees, setup.network, lam_values, gam_values)

    # Training
    coef = gblasso(setup.y_train, setup.x_train, setup.degrees, setup.network, lam, gam)

    return Model(coef, params={"lam":lam, "gam":gam}, from_matlab=False)


def cvGblasso(Y, X, wt, network, lambdas, gammas):
    best_mse = 999999
    kf = KFold(n_splits=folds, shuffle=True, random_state=1)
    for lam in lambdas:
        for gam in gammas:
            errors = []
            for training, holdout in kf.split(X):
                coef = gblasso(Y[training], X[training,:], wt, network, lam, gam)
                errors.append(mean_squared_error(Y[holdout], np.sum(X[holdout] * coef, axis=1)))
            mse = np.mean(errors)
            print("Lambda = %.2f,\t Gamma = %.2f,\t MSE = %.2f" % (lam, gam, mse))
            if mse < best_mse:
                best_mse = mse
                best_lam = lam
                best_gam = gam
    return best_lam, best_gam


def gblasso(Y, X, wt, network, lam, gam):
    b0 = np.zeros(X.shape[1])
    net_pen_mult = lam * (2.0 ** (1.0 - (1.0 / gam)))
    return minimize(gblasso_penalty, b0, (Y, X, wt, network, gam, net_pen_mult), method=minimize_method).x


def gblasso_penalty(b, Y, X, wt, network, gam, net_pen_mult):
    errors = sum(np.power(np.sum(X * b, axis=1) - Y, 2))
    network_penalty = sum([abs(b[i1 - 1]) ** gam / wt[i1 - 1] + abs(b[i2 - 1]) ** gam / wt[i2 - 1]
                           for (i1, i2) in network]) ** (1.0 / gam)
    return errors + network_penalty * net_pen_mult
