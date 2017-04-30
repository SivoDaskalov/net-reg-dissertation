from commons import grace_lambda1_values as lam1, grace_lambda2_values as lam2
from orchestrated_tuning.utilities import get_middle_value, get_middle_index, pack_method_properties, update_parameters
from models.grace import param_fit_grace, param_fit_agrace
from copy import deepcopy


def init_grace_model(setup, matlab_engine):
    method = "grace"
    param_values = {"lam1": lam1, "lam2": lam2}
    cur_params = {"lam1": get_middle_value(lam1), "lam2": get_middle_value(lam2)}
    cur_param_idx = {"lam1": get_middle_index(lam1), "lam2": get_middle_index(lam2)}
    cur_fit = param_fit_grace(setup, matlab_engine, cur_params["lam1"], cur_params["lam2"], use_tuning_set=True)
    cur_coef = cur_fit.coef_
    return pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, tune_grace)


def init_agrace_model(setup, matlab_engine, enet_fit):
    method = "agrace"
    param_values = {"lam1": lam1, "lam2": lam2}
    cur_params = {"lam1": get_middle_value(lam1), "lam2": get_middle_value(lam2)}
    cur_param_idx = {"lam1": get_middle_index(lam1), "lam2": get_middle_index(lam2)}
    cur_fit = param_fit_agrace(setup, matlab_engine, cur_params["lam1"], cur_params["lam2"], enet_fit,
                               use_tuning_set=True)
    cur_coef = cur_fit.coef_
    return pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, tune_grace)


def tune_grace(setup, matlab_engine, methods, method, lam1_idx, lam2_idx):
    local_method = deepcopy(method)
    update_parameters(local_method, {"lam1": lam1_idx, "lam2": lam2_idx})
    local_method["cur_fit"] = param_fit_grace(setup, matlab_engine, local_method["cur_params"]["lam1"],
                                              local_method["cur_params"]["lam2"], use_tuning_set=True)
    local_method["cur_coef"] = local_method["cur_fit"].coef_
    return local_method


def tune_agrace(setup, matlab_engine, methods, method, lam1_idx, lam2_idx):
    enet = None
    for met in methods:
        if met["method"] is "enet":
            enet = met
    local_method = deepcopy(method)
    update_parameters(local_method, {"lam1": lam1_idx, "lam2": lam2_idx})
    local_method["cur_fit"] = param_fit_agrace(setup, matlab_engine, local_method["cur_params"]["lam1"],
                                               local_method["cur_params"]["lam2"], enet["cur_fit"], use_tuning_set=True)
    local_method["cur_coef"] = local_method["cur_fit"].coef_
    return local_method
