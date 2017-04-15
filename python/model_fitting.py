from models.glm import fit_lasso, fit_enet
import time


enable_logging = True
full_method_list = ["lasso", "enet"]


def fit_models(setup, methods = full_method_list):
    models = {}

    if "lasso" in methods:
        log_timestamp(setup.label, "lasso")
        models["lasso"] = fit_lasso(setup)

    if "enet" in methods:
        log_timestamp(setup.label, "enet")
        models["enet"] = fit_enet(setup)

    return models


def batch_fit_models(setups, methods = full_method_list):
    return [(setup, fit_models(setup=setup, methods=methods)) for setup in setups]


def log_timestamp(setup, method):
    if enable_logging:
        print("%s\t%s\t%.0f s" % (setup, method, time.clock()))