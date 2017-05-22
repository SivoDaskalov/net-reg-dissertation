from collections import namedtuple
from datetime import datetime
import pickle

Dataset = namedtuple('Dataset', ['label', 'network', 'degrees', 'methylation', 'expression'])
Setup = namedtuple('Setup', ['label', 'true_coefficients', 'network', 'degrees',
                             'x_tune', 'y_tune', 'x_train', 'y_train', 'x_test', 'y_test'])
full_method_list = ["lasso", "enet", "grace", "agrace", "gblasso", "linf", "alinf", "ttlp", "ltlp", "composite"]
real_data_methods = ["lasso", "enet", "grace", "gblasso", "linf", "composite"]

# General properties
orchestrated_tuning_max_iter = 1000
cv_n_folds = 5
epsilon = 1e-6  # Trim coefficients with absolute value below epsilon

# GLM properties
glm_n_alphas = 100
glm_l1_ratios = [0.1, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .95, .99]
glm_max_iter = 10000

lasso_alpha_opt = 0.01
enet_alpha_opt = 0.04
glm_l1_ratio_opt = 0.9

# Grace properties
grace_lambda1_values = [10.0 ** x for x in range(-2, 5)]
grace_lambda2_values = [10.0 ** x for x in range(-2, 5)]

grace_lambda1_opt = 10.0
grace_lambda2_opt = 10000
agrace_lambda1_opt = 10.0
agrace_lambda2_opt = 0.1

# GBLasso properties
gblasso_lambda_values = [10.0 ** x for x in range(-2, 5)]
gblasso_gamma_values = [2.0, 3.0]
gblasso_maxiter = 1000

gblasso_lambda_opt = 10.0
gblasso_gamma_opt = 3.0

# IMPORTANT!
# Beware of Matlab CVX settings for Linf, aLinf, TTLP and LTLP models, using SeDuMi solver with reduced precision

# Linf properties
linf_c_values = [5.0 * (x + 1) for x in range(20)]
alinf_e_values = [5.0 * (x + 1) for x in range(20)]

linf_c_opt = 35.0
alinf_e_opt = 50.0

# TLP properties
tlp_n_deltas1 = 3
tlp_n_deltas2 = 3
tlp_n_taus = 3
tlp_n_folds = 2  # Greatly reduced number of CV folds due to the duration of calculations
# Use center of calculated intervals for the tuning parameters as optimal tuning parameter values


# Composite model properties
cm_vote_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

cm_vote_thresh_opt = 0.6


def timestamp():
    return "%s >>> " % datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def dump(obj, url):
    with open(url, 'wb') as f:
        pickle.dump(obj, f)


def load(url):
    with open(url, 'rbU') as f:
        return pickle.load(f)


mapping_files = {
    "Lasso_Body": "mappings/body/lasso.csv",
    "Enet_Body": "mappings/body/enet.csv",
    "Grace_Body": "mappings/body/grace.csv",
    "Composite_Body": "mappings/body/composite.csv",
    "Lasso_Prom": "mappings/prom/lasso.csv",
    "Enet_Prom": "mappings/prom/enet.csv",
    "Grace_Prom": "mappings/prom/grace.csv",
    "Composite_Prom": "mappings/prom/composite.csv"
}

error_files = {
    "Body": "mappings/body/errors.csv",
    "Prom": "mappings/prom/errors.csv"
}

fraction_votes_files = {
    "Body": "mappings/body/composite_fraction_votes.csv",
    "Prom": "mappings/prom/composite_fraction_votes.csv"
}
