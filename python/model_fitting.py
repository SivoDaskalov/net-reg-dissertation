from models.glm import fit_lasso, fit_enet
from models.grace import fit_grace, fit_agrace
from models.gblasso import fit_gblasso
import matlab.engine
import os.path
import pickle
import time

enable_logging = False
full_method_list = ["lasso", "enet", "grace", "agrace"]


def fit_models(setup, methods = full_method_list, load_dump = True):
    engine = None
    models = {}

    base_dump_url = "dumps/%s_n%d_p%d_" % (setup.label, setup.x_tune.shape[0], setup.x_tune.shape[1])

    if "agrace" in methods and "enet" not in methods:
        methods.append("enet")

    if "lasso" in methods:
        method = "lasso"
        dump_url = base_dump_url + method
        log_timestamp(setup.label, method)
        if load_dump and os.path.exists(dump_url):
            with open(dump_url, 'rb') as f:
                models[method] = pickle.load(f)
        else:
            models[method] = fit_lasso(setup=setup)
            with open(dump_url, 'wb') as f:
                pickle.dump(models[method], f)

    if "enet" in methods:
        method = "enet"
        dump_url = base_dump_url + method
        log_timestamp(setup.label, method)
        if load_dump and os.path.exists(dump_url):
            with open(dump_url, 'rb') as f:
                models[method] = pickle.load(f)
        else:
            models[method] = fit_enet(setup=setup)
            with open(dump_url, 'wb') as f:
                pickle.dump(models[method], f)

    if "grace" in methods:
        method = "grace"
        dump_url = base_dump_url + method
        log_timestamp(setup.label, method)
        if load_dump and os.path.exists(dump_url):
            with open(dump_url, 'rb') as f:
                models[method] = pickle.load(f)
        else:
            if engine is None:
                engine = start_matlab_engine()
            models[method] = fit_grace(setup=setup, matlab_engine=engine)
            with open(dump_url, 'wb') as f:
                pickle.dump(models[method], f)

    if "agrace" in methods:
        method = "agrace"
        dump_url = base_dump_url + method
        log_timestamp(setup.label, method)
        if load_dump and os.path.exists(dump_url):
            with open(dump_url, 'rb') as f:
                models[method] = pickle.load(f)
        else:
            if engine is None:
                engine = start_matlab_engine()
            models[method] = fit_agrace(setup=setup, matlab_engine=engine, enet_fit=models["enet"])
            with open(dump_url, 'wb') as f:
                pickle.dump(models[method], f)

    if "gblasso" in methods:
        method = "gblasso"
        dump_url = base_dump_url + method
        log_timestamp(setup.label, method)
        if load_dump and os.path.exists(dump_url):
            with open(dump_url, 'rb') as f:
                models[method] = pickle.load(f)
        else:
            if engine is None:
                engine = start_matlab_engine()
            models[method] = fit_gblasso(setup=setup, matlab_engine=engine)
            with open(dump_url, 'wb') as f:
                pickle.dump(models[method], f)

    return models


def batch_fit_models(setups, methods = full_method_list, load_dump = True):
    return [(setup, fit_models(setup=setup, methods=methods, load_dump=load_dump)) for setup in setups]


def log_timestamp(setup, method):
    if enable_logging:
        print("%s\t%s\t%.0f s" % (setup, method, time.clock()))


def start_matlab_engine():
    return matlab.engine.start_matlab("-nodesktop")