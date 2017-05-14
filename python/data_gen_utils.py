from commons import Setup, timestamp
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
import os.path
import pickle


# As suggested by Li and Li, Bioinformatics 2008, section 4
def generate_true_coefficients(trans_factor_coefficients, regulated_denominator, n_regulated_negative,
                               n_regulated_positive, n_trailing_zero_genes):
    leading_coefficients = [[tf] + [-tf / regulated_denominator] * n_regulated_negative +
                            [tf / regulated_denominator] * n_regulated_positive for tf in trans_factor_coefficients]
    true_coefficients = [coef for sublist in leading_coefficients for coef in sublist] + [0] * n_trailing_zero_genes
    return true_coefficients


def generate_setup_coefficients(n_trans_factors):
    # 10 regulated genes per trans factor, setups are as suggested by Li and Li, Bioinformatics 2008, section 4
    trans_factor_coefs = [5, -5, 3, -3]
    n_trailing_zero_genes = (n_trans_factors - len(trans_factor_coefs)) * 11
    setups = {
        "Setup 1": generate_true_coefficients(trans_factor_coefs, math.sqrt(10), 0, 10, n_trailing_zero_genes),
        "Setup 2": generate_true_coefficients(trans_factor_coefs, math.sqrt(10), 3, 7, n_trailing_zero_genes),
        "Setup 3": generate_true_coefficients(trans_factor_coefs, 10, 0, 10, n_trailing_zero_genes),
        "Setup 4": generate_true_coefficients(trans_factor_coefs, 10, 3, 7, n_trailing_zero_genes)
    }
    return setups


def generate_observation(n_trans_factors, n_regulated_genes_per_trans_factor):
    tf_expression_levels = np.random.normal(loc=0.0, scale=1.1, size=n_trans_factors)
    expressions = [
        [tf] + np.random.normal(loc=0.7 * tf, scale=math.sqrt(0.51), size=n_regulated_genes_per_trans_factor).tolist()
        for tf in tf_expression_levels]
    observation = [item for sublist in expressions for item in sublist]
    return observation


def generate_expressions(n_observations, n_trans_factors, n_regulated_genes_per_trans_factor):
    expressions = np.empty(shape=(n_observations, n_trans_factors * (n_regulated_genes_per_trans_factor + 1)))
    for i in range(n_observations):
        expressions[i] = generate_observation(n_trans_factors, n_regulated_genes_per_trans_factor)
    normalized_expressions = StandardScaler().fit_transform(expressions)
    return normalized_expressions


def generate_response(expressions, coefficients):
    response = np.sum(expressions * coefficients, axis=1)
    noise = np.random.normal(loc=0, scale=math.sqrt(np.var(coefficients)), size=response.shape[0])
    noisy_response = [response[i] + noise[i] for i in range(response.shape[0])]
    normalized_response = noisy_response - np.mean(noisy_response)
    return normalized_response


def generate_network(n_trans_factors, n_regulated_genes_per_trans_factor):
    network = [(i * (n_regulated_genes_per_trans_factor + 1) + 1, i * (n_regulated_genes_per_trans_factor + 1) + j + 1)
               for i in range(n_trans_factors) for j in range(1, n_regulated_genes_per_trans_factor + 1)]
    degrees = np.tile([n_regulated_genes_per_trans_factor] + [1] * n_regulated_genes_per_trans_factor, n_trans_factors)
    return network, degrees


def batch_generate_setups(n_trans_factors, n_regulated_genes_per_trans_factor,
                          n_tune_obs, n_train_obs, n_test_obs, load_dump=False):
    dump_url = "dumps/setups_tf%d_rg%d_tu%d_tr%d_ts%d" % (n_trans_factors, n_regulated_genes_per_trans_factor,
                                                          n_tune_obs, n_train_obs, n_test_obs)
    if load_dump and os.path.exists(dump_url):
        print("%sLoading previously generated dataset" % timestamp())
        with open(dump_url, 'rbU') as f:
            setups = pickle.load(f)
        print("%sLoaded previously generated dataset" % timestamp())
    else:
        print("Generating dataset")
        setups = []
        network, degrees = generate_network(n_trans_factors, n_regulated_genes_per_trans_factor)
        for label, coefficients in generate_setup_coefficients(n_trans_factors).items():
            x_tune = generate_expressions(n_tune_obs, n_trans_factors, n_regulated_genes_per_trans_factor)
            y_tu = generate_response(x_tune, coefficients)

            x_train = generate_expressions(n_train_obs, n_trans_factors, n_regulated_genes_per_trans_factor)
            y_tr = generate_response(x_train, coefficients)

            x_test = generate_expressions(n_test_obs, n_trans_factors, n_regulated_genes_per_trans_factor)
            y_ts = generate_response(x_test, coefficients)

            setups.append(Setup(label=label, true_coefficients=coefficients, network=network, degrees=degrees,
                                x_tune=x_tune, y_tune=y_tu, x_train=x_train, y_train=y_tr, x_test=x_test, y_test=y_ts))
        with open(dump_url, 'wb') as f:
            pickle.dump(setups, f)
    return setups
