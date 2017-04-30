from glm import init_enet_model, init_lasso_model
from grace import init_grace_model, init_agrace_model
from gblasso import init_gblasso_model
from linf import init_linf_model, init_alinf_model
from models.glm import param_fit_lasso, param_fit_enet
from models.grace import param_fit_grace, param_fit_agrace
from models.gblasso import param_fit_gblasso
from models.linf import param_fit_linf, param_fit_alinf
from utilities import get_cache_key
from commons import orchestrated_tuning_max_iter as max_iter
import matlab.engine
import numpy as np
import itertools
import os.path
import pickle

methods = ["lasso", "enet", "grace", "agrace", "gblasso", "linf", "alinf"]


def init_methods(setup, matlab_engine):
    np.seterr(divide='ignore', invalid='ignore')
    lasso = init_lasso_model(setup)
    enet = init_enet_model(setup)
    grace = init_grace_model(setup, matlab_engine)
    agrace = init_agrace_model(setup, matlab_engine, enet["cur_fit"])
    gblasso = init_gblasso_model(setup)
    linf = init_linf_model(setup, matlab_engine)
    alinf = init_alinf_model(setup, matlab_engine, linf["cur_fit"])
    return [lasso, enet, grace, agrace, gblasso, linf, alinf]


def train_methods(setup, matlab_engine, reference_methods):
    models = {}
    for name, reference in reference_methods.iteritems():
        if name is "lasso":
            models[name] = param_fit_lasso(setup, *reference["cur_params"].values())
        if name is "enet":
            models[name] = param_fit_enet(setup, *reference["cur_params"].values())
        if name is "grace":
            models[name] = param_fit_grace(setup, matlab_engine, *reference["cur_params"].values())
        if name is "agrace":
            models[name] = param_fit_agrace(setup, matlab_engine,
                                            *(reference["cur_params"].values() + [models["enet"]]))
        if name is "gblasso":
            models[name] = param_fit_gblasso(setup, *reference["cur_params"].values())
        if name is "linf":
            models[name] = param_fit_linf(setup, matlab_engine, *reference["cur_params"].values())
        if name is "alinf":
            models[name] = param_fit_alinf(setup, matlab_engine, *(reference["cur_params"].values() + [models["linf"]]))
    return models


def tune_abstract_method(setup, matlab_engine, methods, load_dump, cache, method_idx):
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
        if load_dump and model_key in cache[method_name].keys():
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

    dump_url = "dumps/cache/%s_p%d" % (setup.label, setup.x_tune.shape[1])
    if load_dump and os.path.exists(dump_url):
        print("Loaded cache")
        with open(dump_url, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}

    for method in reference_methods:
        cache[method["method"]] = {get_cache_key(method["method"], method["cur_param_idx"].values()): method}

    iter = 0
    converged = False
    while not converged and iter < max_iter:
        iter += 1
        current_methods = [tune_abstract_method(setup, matlab_engine, reference_methods, load_dump, cache, i)
                           for i in range(len(reference_methods))]
        converged = np.all(
            [utilities.compare_params(reference_methods[i], current_methods[i]) for i in range(len(reference_methods))])
        reference_methods = current_methods

    if iter < max_iter:
        print("Orchestrated tuning converged after %d iterations" % iter)
    else:
        print("Orchestrated tuning not converged after max iterations (%d)" % iter)
    with open(dump_url, 'wb') as f:
        pickle.dump(cache, f)
    return train_methods(setup, matlab_engine, {method["method"]: method for method in reference_methods})


def batch_do_orchestrated_tuning(setups, load_dump=True):
    engine = matlab.engine.start_matlab("-nodesktop")
    return [(setup, do_orchestrated_tuning(setup, engine, methods, load_dump)) for setup in setups]
