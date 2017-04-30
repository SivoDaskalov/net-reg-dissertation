from glm import init_enet_model, init_lasso_model, tune_enet, tune_lasso
from utilities import get_cache_key
import matlab.engine
import numpy as np
import itertools
from models.glm import param_fit_lasso, param_fit_enet
from models.grace import param_fit_grace, param_fit_agrace
from models.gblasso import param_fit_gblasso
from models.linf import param_fit_linf, param_fit_alinf

methods = ["lasso", "enet", "grace", "agrace", "gblasso", "linf", "alinf"]


def init_methods(setup, matlab_engine):
    return [init_lasso_model(setup), init_enet_model(setup)]


def train_methods(setup, matlab_engine, reference_methods):
    models = {}
    for name, reference in reference_methods.iteritems():
        if name is "lasso":
            models[name] = param_fit_lasso(setup, *reference["cur_params"].values())
        if name is "enet":
            models[name] = param_fit_enet(setup, *reference["cur_params"].values())
    return models


def tune_abstract_method(setup, matlab_engine, methods, cache, method_idx):
    method = methods[method_idx]
    current_param_indices = method["cur_param_idx"].values()
    method_param_lengths = [len(values) for values in method["param_values"].values()]
    possible_param_indices = [[i - 1, i, i + 1] for i in current_param_indices]
    actual_param_indices = [
        [idx for idx in possible_param_indices[i] if idx >= 0 and idx < method_param_lengths[i]]
        for i in range(len(method_param_lengths))]
    param_combinations = list(itertools.product(*actual_param_indices))

    new_fits = []
    method_name = method["method"]
    for params in param_combinations:
        model_key = get_cache_key(method_name, params)
        if model_key in cache[method_name].keys():
            # print("Cache hit for %s with param indices (%s)" % (method_name, ', '.join(str(param) for param in params)))
            new_fits.append(cache[method_name][model_key])
        else:
            # print("Fitting %s with param indices (%s)" % (method_name, ', '.join(str(param) for param in params)))
            fit = method["callable"](setup, matlab_engine, methods, method, *params)
            cache[method_name][model_key] = fit
            new_fits.append(fit)

    target_coef = np.mean(np.array([method["cur_coef"] for idx, method in enumerate(methods) if idx != method_idx]),
                          axis=0)
    coef_correlations = [np.corrcoef(new_fits[i]["cur_coef"], target_coef)[0, 1] for i in range(len(new_fits))]
    return new_fits[coef_correlations.index(min(coef_correlations))]


def do_orchestrated_tuning(setup, matlab_engine, method_names, load_dump=True):
    print("Orchestrated tuning for %s" % setup.label)
    reference_methods = init_methods(setup, matlab_engine)
    cache = {}
    for method in reference_methods:
        cache[method["method"]] = {get_cache_key(method["method"], method["cur_param_idx"].values()): method}

    iter = 0
    converged = False
    while not converged:
        iter += 1
        current_methods = [tune_abstract_method(setup, matlab_engine, reference_methods, cache, i) for i in
                           range(len(reference_methods))]
        converged = np.all(
            [utilities.compare_params(reference_methods[i], current_methods[i]) for i in range(len(reference_methods))])
        reference_methods = current_methods

    print("Orchestrated tuning converged after %d iterations" % iter)
    return train_methods(setup, matlab_engine, {method["method"]: method for method in reference_methods})


def batch_do_orchestrated_tuning(setups, load_dump=True):
    engine = matlab.engine.start_matlab("-nodesktop")
    return [(setup, do_orchestrated_tuning(setup, engine, methods, load_dump)) for setup in setups]
