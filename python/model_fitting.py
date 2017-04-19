from models.glm import fit_lasso, fit_enet
from models.grace import fit_grace, fit_agrace
from models.gblasso import fit_gblasso
from models.linf import fit_linf, fit_alinf
from models.tlp import fit_tlpi
import matlab.engine
import os.path
import pickle
import time

enable_logging = True
full_method_list = ["lasso", "enet", "grace", "agrace", "gblasso", "linf", "alinf"]


def fit_or_load(setup, method_name, load_dump, fitting_func, args):
    base_dump_url = "dumps/%s_n%d_p%d_" % (setup.label, setup.x_tune.shape[0], setup.x_tune.shape[1])
    dump_url = base_dump_url + method_name
    if load_dump and os.path.exists(dump_url):
        print("Loaded %s model for %s" % (method_name, setup.label))
        with open(dump_url, 'rb') as f:
            fit = pickle.load(f)
    else:
        print("Fitting %s model for %s" % (method_name, setup.label))
        t_ = time.clock()
        fit = fitting_func(setup, *args)
        print("Fitting %s model for %s took %.0f seconds\n" % (method_name, setup.label, time.clock() - t_))
        with open(dump_url, 'wb') as f:
            pickle.dump(fit, f)
    return fit


def fit_models(setup, engine, methods=full_method_list, load_dump=True):
    models = {}

    if "agrace" in methods and "enet" not in methods:
        methods.append("enet")
    if "alinf" in methods and "linf" not in methods:
        methods.append("linf")
    if "tlpi" in methods and "lasso" not in methods:
        methods.append("lasso")

    if "lasso" in methods:
        method = "lasso"
        models[method] = fit_or_load(setup, method, load_dump, fit_lasso, [])

    if "enet" in methods:
        method = "enet"
        models[method] = fit_or_load(setup, method, load_dump, fit_enet, [])

    if "grace" in methods:
        method = "grace"
        models[method] = fit_or_load(setup, method, load_dump, fit_grace, [engine])

    if "agrace" in methods:
        method = "agrace"
        models[method] = fit_or_load(setup, method, load_dump, fit_agrace, [engine, models["enet"]])

    if "gblasso" in methods:
        method = "gblasso"
        models[method] = fit_or_load(setup, method, load_dump, fit_gblasso, [])

    if "linf" in methods:
        method = "linf"
        models[method] = fit_or_load(setup, method, load_dump, fit_linf, [engine])

    if "alinf" in methods:
        method = "alinf"
        models[method] = fit_or_load(setup, method, load_dump, fit_alinf, [engine, models["linf"]])

    if "tlpi" in methods:
        method = "tlpi"
        models[method] = fit_or_load(setup, method, load_dump, fit_tlpi, [engine, models["lasso"]])

    return models


def batch_fit_models(setups, methods=full_method_list, load_dump=True):
    engine = matlab.engine.start_matlab("-nodesktop")
    return [(setup, fit_models(setup=setup, engine=engine, methods=methods, load_dump=load_dump)) for setup in setups]
