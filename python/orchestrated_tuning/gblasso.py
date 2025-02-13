from commons import gblasso_lambda_values as lam, gblasso_gamma_values as gam
from orchestrated_tuning.utilities import get_initial_param, pack_method_properties, update_parameters
from models.gblasso import param_fit_gblasso
from copy import deepcopy


def init_gblasso_model(setup, custom_start_point=False):
    method = "gblasso"
    param_values = {"gamma": gam, "lambda": lam}
    gam_idx, gam_val = get_initial_param(grid=gam, setup=setup.label, method="gblasso", param_name="gamma")
    lam_idx, lam_val = get_initial_param(grid=lam, setup=setup.label, method="gblasso", param_name="lambda")
    cur_params = {"gamma": gam_val, "lambda": lam_val}
    cur_param_idx = {"gamma": gam_idx, "lambda": lam_idx}
    cur_fit = param_fit_gblasso(setup, cur_params["lambda"], cur_params["gamma"], use_tuning_set=True)
    cur_coef = cur_fit.coef_
    return pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, tune_gblasso)


def tune_gblasso(setup, matlab_engine, methods, method, gamma_idx, lambda_idx):
    local_method = deepcopy(method)
    update_parameters(local_method, {"lambda": lambda_idx, "gamma": gamma_idx})
    local_method["cur_fit"] = param_fit_gblasso(setup, local_method["cur_params"]["lambda"],
                                                local_method["cur_params"]["gamma"], use_tuning_set=True)
    local_method["cur_coef"] = local_method["cur_fit"].coef_
    return local_method
