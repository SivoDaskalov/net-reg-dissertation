from __future__ import division
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


def evaluate_model(setup, model):
    y_true = setup.y_test
    y_pred = model.predict(setup.x_test)
    true_coef = setup.true_coefficients
    estimated_coef = model.coef_

    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    correlation = np.corrcoef(true_coef, estimated_coef)[0, 1]
    n_predictors = np.count_nonzero(estimated_coef)

    true_predictors = np.count_nonzero(true_coef)
    false_positive_coef = np.count_nonzero(estimated_coef[true_predictors:])

    sensitivity = np.count_nonzero(estimated_coef[:true_predictors]) / true_predictors
    specificity = (len(true_coef) - true_predictors - false_positive_coef) / (len(true_coef) - true_predictors)
    precision = np.count_nonzero(estimated_coef[:true_predictors]) / np.count_nonzero(estimated_coef)

    return [mse, n_predictors, correlation, sensitivity, specificity, precision]


def batch_evaluate_models(fits):
    result_fields = ["setup", "model", "mse", "predictors", "correlation", "sens", "spec", "prec", "params"]
    results = []

    for (setup, models) in fits:
        for model_name, model in models.items():
            mse, n_predictors, corr, sens, spec, prec = evaluate_model(setup, model)
            params = ', '.join(['{}={}'.format(k,v) for k,v in model.params_.iteritems()])
            new_row = [setup.label, model_name, mse, n_predictors, corr, sens, spec, prec, params]
            results = np.append(results, new_row)

    results.shape = (int(results.shape[0] / len(result_fields)), len(result_fields))
    results = pd.DataFrame(data=results, columns=result_fields)
    return results
