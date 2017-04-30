from __future__ import division
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


def evaluate_model(setup, model):
    y_true = setup.y_test
    y_pred = model.predict(setup.x_test)
    estimated_coef = model.coef_

    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    n_predictors = np.count_nonzero(estimated_coef)

    if setup.true_coefficients is None:
        correlation = sensitivity = specificity = precision = None
    else:
        true_coef = setup.true_coefficients
        correlation = np.corrcoef(true_coef, estimated_coef)[0, 1]
        true_predictors = np.count_nonzero(true_coef)
        false_positive_coef = np.count_nonzero(estimated_coef[true_predictors:])

        sensitivity = np.count_nonzero(estimated_coef[:true_predictors]) / true_predictors
        specificity = (len(true_coef) - true_predictors - false_positive_coef) / (len(true_coef) - true_predictors)
        precision = np.count_nonzero(estimated_coef[:true_predictors]) / np.count_nonzero(estimated_coef)

    return [mse, n_predictors, correlation, sensitivity, specificity, precision]


def batch_evaluate_models(fits, filename=None):
    result_fields = ["setup", "model", "mse", "predictors", "correlation", "sens", "spec", "prec", "params"]
    results = []

    for (setup, models) in fits:
        for model_name, model in models.items():
            mse, n_predictors, corr, sens, spec, prec = evaluate_model(setup, model)
            params = ', '.join(['{}={}'.format(k, v) for k, v in model.params_.iteritems()])
            new_row = [setup.label, model_name, mse, n_predictors, corr, sens, spec, prec, params]
            results = np.append(results, new_row)

    results.shape = (int(results.shape[0] / len(result_fields)), len(result_fields))
    results = pd.DataFrame(data=results, columns=result_fields)
    results = results.sort_values(['model', 'setup'])
    if filename is None:
        filename = "results/p%d.csv" % fits[0][0].x_test.shape[1]
    results.to_csv(filename, sep=',')
    return results


def load_results_from_csv(p):
    return pd.read_csv("results/p%d.csv" % p, sep=',', index_col=0)
