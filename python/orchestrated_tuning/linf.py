from commons import linf_c_values as c, alinf_e_values as e
from orchestrated_tuning.utilities import get_initial_param, pack_method_properties, update_parameters
from models.linf import param_fit_linf, param_fit_alinf
from copy import deepcopy


def init_linf_model(setup, matlab_engine):
    method = "linf"
    param_values = {"c": c}
    c_idx, c_val = get_initial_param(grid=c, setup=setup.label, method="linf", param_name="C")
    cur_params = {"c": c_val}
    cur_param_idx = {"c": c_idx}
    cur_fit = param_fit_linf(setup, matlab_engine, cur_params["c"], use_tuning_set=True)
    cur_coef = cur_fit.coef_
    return pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, tune_linf)


def init_alinf_model(setup, matlab_engine, linf_fit):
    method = "linf"
    param_values = {"e": e}
    e_idx, e_val = get_initial_param(grid=e, setup=setup.label, method="alinf", param_name="E")
    cur_params = {"e": e_val}
    cur_param_idx = {"e": e_idx}
    cur_fit = param_fit_alinf(setup, matlab_engine, cur_params["e"], linf_fit, use_tuning_set=True)
    cur_coef = cur_fit.coef_
    return pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, tune_linf)


def tune_linf(setup, matlab_engine, methods, method, c_idx):
    local_method = deepcopy(method)
    update_parameters(local_method, {"c": c_idx})
    local_method["cur_fit"] = param_fit_linf(setup, matlab_engine, local_method["cur_params"]["c"], use_tuning_set=True)
    local_method["cur_coef"] = local_method["cur_fit"].coef_
    return local_method


def tune_alinf(setup, matlab_engine, methods, method, e_idx):
    linf = None
    for met in methods:
        if met["method"] is "linf":
            linf = met
    local_method = deepcopy(method)
    update_parameters(local_method, {"e": e_idx})
    local_method["cur_fit"] = param_fit_alinf(setup, matlab_engine, local_method["cur_params"]["c"], linf["cur_fit"],
                                              use_tuning_set=True)
    local_method["cur_coef"] = local_method["cur_fit"].coef_
    return local_method
