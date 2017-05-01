from glm import init_enet_model, init_lasso_model
from grace import init_grace_model, init_agrace_model
from gblasso import init_gblasso_model
from linf import init_linf_model, init_alinf_model
from models.glm import param_fit_lasso, param_fit_enet
from models.grace import param_fit_grace, param_fit_agrace
from models.gblasso import param_fit_gblasso
from models.linf import param_fit_linf, param_fit_alinf
from utilities import get_cache_key
from commons import orchestrated_tuning_max_iter as max_iter, timestamp
import matlab.engine
import numpy as np
import itertools
import os.path
import pickle
import math

methods = ["lasso", "enet", "grace", "gblasso", "linf"]
always_recalculate = ["agrace", "alinf"]
optimization_methods = ["coef_correlation", "n_predictors"]
optimization_method = "coef_correlation"
use_tuning_set_for_final_training = False


def init_methods(setup, matlab_engine, method_names, load_dump):
    np.seterr(divide='ignore', invalid='ignore')
    dump_url = "dumps/cache/initial_%s_p%d" % (setup.label, setup.x_tune.shape[1])
    if load_dump and os.path.exists(dump_url):
        print("%sLoading initial method models" % timestamp())
        with open(dump_url, 'rb') as f:
            methods = pickle.load(f)
        print("%sLoaded initial method models" % timestamp())
    else:
        methods = []
        if "lasso" in method_names:
            methods.append(init_lasso_model(setup))
        if "enet" in method_names:
            enet = init_enet_model(setup)
            methods.append(enet)
        if "grace" in method_names:
            methods.append(init_grace_model(setup, matlab_engine))
        if "agrace" in method_names:
            methods.append(init_agrace_model(setup, matlab_engine, enet["cur_fit"]))
        if "gblasso" in method_names:
            methods.append(init_gblasso_model(setup))
        if "linf" in method_names:
            linf = init_linf_model(setup, matlab_engine)
            methods.append(linf)
        if "alinf" in method_names:
            methods.append(init_alinf_model(setup, matlab_engine, linf["cur_fit"]))
        with open(dump_url, 'wb') as f:
            pickle.dump(methods, f)
    return methods


def init_cache(setup, reference_methods, load_dump):
    dump_url = "dumps/cache/models_%s_p%d" % (setup.label, setup.x_tune.shape[1])
    if load_dump and os.path.exists(dump_url):
        print("%sLoading model cache" % timestamp())
        with open(dump_url, 'rb') as f:
            cache = pickle.load(f)
        print("%sLoaded model cache" % timestamp())
    else:
        cache = {}
        for method in reference_methods:
            cache[method["method"]] = {get_cache_key(method["method"], method["cur_param_idx"].values()): method}
    return cache, dump_url


def train_methods(setup, matlab_engine, reference_methods):
    models = {}
    use_tuning_set = use_tuning_set_for_final_training
    print("%sTraining methods with tuned parameters" % timestamp())
    for name, reference in reference_methods.iteritems():
        params = reference["cur_params"]
        if name == "lasso":
            models[name] = param_fit_lasso(setup, params["alpha"], use_tuning_set)
        if name == "enet":
            models[name] = param_fit_enet(setup, params["alpha"], params["l1_ratio"], use_tuning_set)
        if name == "grace":
            models[name] = param_fit_grace(setup, matlab_engine, params["lam1"], params["lam2"], use_tuning_set)
        if name == "agrace":
            models[name] = param_fit_agrace(setup, matlab_engine, params["lam1"], params["lam2"],
                                            models["enet"], use_tuning_set)
        if name == "gblasso":
            models[name] = param_fit_gblasso(setup, params["lambda"], params["gamma"], use_tuning_set)
        if name == "linf":
            models[name] = param_fit_linf(setup, matlab_engine, params["c"], use_tuning_set)
        if name == "alinf":
            models[name] = param_fit_alinf(setup, matlab_engine, params["e"], models["linf"], use_tuning_set)
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
        if method_name not in always_recalculate and model_key in cache[method_name].keys():
            # print("%sCache hit for %s with param indices (%s)" % (
            #     timestamp(), method_name, ', '.join(str(param) for param in params)))
            new_fits.append(cache[method_name][model_key])
        else:
            # print("%sFitting %s with param indices (%s)" % (
            #     timestamp(), method_name, ', '.join(str(param) for param in params)))
            fit = method["callable"](setup, matlab_engine, methods, method, *params)
            cache[method_name][model_key] = fit
            new_fits.append(fit)

    if optimization_method is "coef_correlation":
        target_coef = np.mean(np.array([method["cur_coef"] for idx, method in enumerate(methods) if idx != method_idx]),
                              axis=0)
        coef_correlations = [np.corrcoef(new_fits[i]["cur_coef"], target_coef)[0, 1] for i in range(len(new_fits))]
        opt_index = coef_correlations.index(max(coef_correlations))

    if optimization_method is "n_predictors":
        n_predictors_target = math.ceil(np.mean(
            np.count_nonzero([method["cur_coef"] for idx, method in enumerate(methods) if idx != method_idx], axis=1)))
        n_predictors_cur = [np.count_nonzero(new_fits[i]["cur_coef"]) for i in range(len(new_fits))]
        opt_n_predictors = min(n_predictors_cur, key=lambda x: abs(x - n_predictors_target))
        candidate_indices = [idx for idx, n_pred in enumerate(n_predictors_cur) if n_pred == opt_n_predictors]

        if len(candidate_indices) > 1:
            target_coef = np.mean(
                np.array([method["cur_coef"] for idx, method in enumerate(methods) if idx != method_idx]),
                axis=0)
            coef_correlations = [np.corrcoef(new_fits[i]["cur_coef"], target_coef)[0, 1] for i in candidate_indices]
            opt_index = coef_correlations.index(max(coef_correlations))
        else:
            opt_index = candidate_indices[0]

    return new_fits[opt_index]


def do_orchestrated_tuning(setup, matlab_engine, method_names, load_dump=True):
    print("%sOrchestrated tuning for %s" % (timestamp(), setup.label))
    reference_methods = init_methods(setup, matlab_engine, method_names, load_dump)
    cache, dump_url = init_cache(setup, reference_methods, load_dump)

    iter = 0
    converged = False
    while not converged and iter < max_iter:
        iter += 1
        # if iter % 100 == 0:
        print("%sIteration %d" % (timestamp(), iter))
        current_methods = [tune_abstract_method(setup, matlab_engine, reference_methods, cache, i)
                           for i in range(len(reference_methods))]
        converged = np.all(
            [utilities.compare_params(reference_methods[i], current_methods[i]) for i in range(len(reference_methods))])
        reference_methods = current_methods

    if iter < max_iter:
        print("%sOrchestrated tuning converged after %d iterations" % (timestamp(), iter))
    else:
        print("%sOrchestrated tuning not converged after max iterations (%d)" % (timestamp(), iter))
    with open(dump_url, 'wb') as f:
        pickle.dump(cache, f)

    return train_methods(setup, matlab_engine, {method["method"]: method for method in reference_methods})


def batch_do_orchestrated_tuning(setups, load_dump=True):
    engine = matlab.engine.start_matlab("-nodesktop")
    return [(setup, do_orchestrated_tuning(setup, engine, methods, load_dump)) for setup in setups]
