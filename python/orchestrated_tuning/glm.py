from commons import glm_l1_ratios, glm_n_alphas, glm_max_iter, cv_n_folds
from sklearn.linear_model import LassoCV, ElasticNetCV
from orchestrated_tuning.utilities import get_middle_value, get_middle_index, pack_method_properties, update_parameters
from models.glm import param_fit_lasso, param_fit_enet
from copy import deepcopy


def init_lasso_model(setup):
    model = LassoCV(n_alphas=glm_n_alphas, cv=cv_n_folds, n_jobs=-1, max_iter=glm_max_iter,
                    random_state=1, fit_intercept=False).fit(X=setup.x_tune, y=setup.y_tune)
    alphas = sorted(model.alphas_)
    method = "lasso"
    param_values = {"alpha": alphas}
    cur_params = {"alpha": get_middle_value(alphas)}
    cur_param_idx = {"alpha": get_middle_index(alphas)}
    cur_fit = param_fit_lasso(setup, cur_params["alpha"], use_tuning_set=True)
    cur_coef = cur_fit.coef_
    return pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, tune_lasso)


def init_enet_model(setup):
    model = ElasticNetCV(n_alphas=glm_n_alphas, l1_ratio=glm_l1_ratios, cv=cv_n_folds, n_jobs=-1, max_iter=glm_max_iter,
                         random_state=1, fit_intercept=False).fit(X=setup.x_tune, y=setup.y_tune)
    alphas = sorted([item for sublist in model.alphas_.tolist() for item in sublist])
    method = "enet"
    param_values = {"alpha": alphas, "l1_ratio": glm_l1_ratios}
    cur_params = {"alpha": get_middle_value(alphas), "l1_ratio": get_middle_value(glm_l1_ratios)}
    cur_param_idx = {"alpha": get_middle_index(alphas), "l1_ratio": get_middle_index(glm_l1_ratios)}
    cur_fit = param_fit_enet(setup, cur_params["alpha"], cur_params["l1_ratio"], use_tuning_set=True)
    cur_coef = cur_fit.coef_
    return pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, tune_enet)


def tune_lasso(setup, matlab_engine, methods, method, alpha_idx):
    local_method = deepcopy(method)
    update_parameters(local_method, {"alpha": alpha_idx})
    local_method["cur_fit"] = param_fit_lasso(setup, local_method["cur_params"]["alpha"], use_tuning_set=True)
    local_method["cur_coef"] = local_method["cur_fit"].coef_
    return local_method


def tune_enet(setup, matlab_engine, methods, method, alpha_idx, l1_ratio_idx):
    local_method = deepcopy(method)
    update_parameters(local_method, {"alpha": alpha_idx, "l1_ratio": l1_ratio_idx})
    local_method["cur_fit"] = param_fit_enet(setup, local_method["cur_params"]["alpha"],
                                              local_method["cur_params"]["l1_ratio"], use_tuning_set=True)
    local_method["cur_coef"] = local_method["cur_fit"].coef_
    return local_method
