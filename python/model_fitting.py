from models.glm import fit_lasso, fit_enet
from commons import Setup
import time


full_method_list = ["lasso", "enet"]


def fit_models(setup: Setup, methods: list = full_method_list):
    models = {}

    if "lasso" in methods:
        log_timestamp(setup.label, "lasso")
        models["lasso"] = fit_lasso(setup)

    if "enet" in methods:
        log_timestamp(setup.label, "enet")
        models["enet"] = fit_enet(setup)

    return models


def batch_fit_models(setups: list, methods: list = full_method_list) -> list:
    return [(setup, fit_models(setup=setup, methods=methods)) for setup in setups]


def log_timestamp(setup: str, method: str):
    print("%s\t%s\t%.0f s" % (setup, method, time.process_time()))