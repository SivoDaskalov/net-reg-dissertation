from commons import glm_l1_ratios, glm_n_alphas, glm_max_iter, grace_lambda1_values, grace_lambda2_values, \
    gblasso_gamma_values, gblasso_lambda_values, gblasso_maxiter, linf_c_values, alinf_e_values, cv_n_folds
from models.glm import param_fit_lasso, param_fit_enet
from models.grace import param_fit_grace, param_fit_agrace
from models.gblasso import param_fit_gblasso
from models.linf import param_fit_linf, param_fit_alinf
from sklearn.linear_model import LassoCV, ElasticNetCV
import matlab.engine

methods = ["enet", "grace", "agrace", "gblasso", "linf", "alinf"]


def get_middle_index(values):
    return (len(values) - 1) / 2


def get_middle_value(values):
    return values[get_middle_index(values)]


def init_lasso_model(setup):
    model = LassoCV(n_alphas=glm_n_alphas, l1_ratio=glm_l1_ratios, cv=cv_n_folds, n_jobs=-1, max_iter=glm_max_iter,
                    random_state=1, fit_intercept=False).fit(X=setup.x_tune, y=setup.y_tune)
    alphas = sorted([item for sublist in model.alphas_.tolist() for item in sublist])

    method = "lasso"
    param_values = {
        "alpha": alphas
    }
    cur_params = {
        "alpha": get_middle_value(alphas)
    }
    cur_param_idx = {
        "alpha": get_middle_index(alphas)
    }
    cur_fit = param_fit_lasso(setup, cur_params["alpha"], use_tuning_set=True)
    cur_coef = cur_fit.coef_

    return pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, tune_lasso)


def init_enet_model(setup):
    model = ElasticNetCV(n_alphas=glm_n_alphas, l1_ratio=glm_l1_ratios, cv=cv_n_folds, n_jobs=-1, max_iter=glm_max_iter,
                         random_state=1, fit_intercept=False).fit(X=setup.x_tune, y=setup.y_tune)
    alphas = sorted([item for sublist in model.alphas_.tolist() for item in sublist])

    method = "enet"
    param_values = {
        "l1_ratio": glm_l1_ratios,
        "alpha": alphas
    }
    cur_params = {
        "l1_ratio": get_middle_value(glm_l1_ratios),
        "alpha": get_middle_value(alphas)
    }
    cur_param_idx = {
        "l1_ratio": get_middle_index(glm_l1_ratios),
        "alpha": get_middle_index(alphas)
    }
    cur_fit = param_fit_enet(setup, cur_params["alpha"], cur_params["l1_ratio"], use_tuning_set=True)
    cur_coef = cur_fit.coef_

    return pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, tune_enet)


def pack_method_properties(method, param_values, cur_params, cur_param_idx, cur_fit, cur_coef, callable):
    return {
        "method": method,
        "param_values": param_values,
        "cur_params": cur_params,
        "cur_param_idx": cur_param_idx,
        "cur_fit": cur_fit,
        "cur_coef": cur_coef,
        "callable": callable
    }


def init_models(setup, matlab_engine):
    methods = []
    methods.append(init_lasso_model(setup))
    methods.append(init_enet_model(setup))
    return methods


def tune_enet(setup, matlab_engine, methods):
    return None


def tune_lasso(setup, matlab_engine, methods):
    return None


def tune_abstract_method(setup, matlab_engine, method, methods, cache):
    return None


def do_orchestrated_tuning(setup, matlab_engine, method_names, load_dump=True):
    models = init_models(setup, matlab_engine)
    cache = {}
    for model in models:
        cache[model.method] = {}

    current_models = []

    return None


def batch_do_orchestrated_tuning(setups, load_dump=True):
    engine = matlab.engine.start_matlab("-nodesktop")
    return [(setup, do_orchestrated_tuning(setup, engine, methods, load_dump)) for setup in setups]
