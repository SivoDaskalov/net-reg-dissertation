from commons import glm_l1_ratios, glm_n_alphas, glm_max_iter, grace_lambda1_values, grace_lambda2_values, \
    gblasso_gamma_values, gblasso_lambda_values, gblasso_maxiter, linf_c_values, alinf_e_values
import matlab.engine

methods = ["enet", "grace", "agrace", "gblasso", "linf", "alinf"]
iterations = 10


def do_orchestrated_tuning(setup, matlab_engine, iterations, method_names, load_dump=True):
    return None


def batch_do_orchestrated_tuning(setups, load_dump=True):
    engine = matlab.engine.start_matlab("-nodesktop")
    return [(setup, do_orchestrated_tuning(setup, engine, iterations, methods, load_dump)) for setup in setups]
