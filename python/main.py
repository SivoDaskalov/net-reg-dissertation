import data_gen_utils as gen
import model_fitting as fitting
import model_metrics as metrics
import time
time.clock()
n_trans_factors = 50
n_regulated_genes_per_trans_factor = 10

setups = gen.batch_generate_setups(n_regulated_genes_per_trans_factor=n_regulated_genes_per_trans_factor,
                                   n_trans_factors=n_trans_factors, n_tune_obs=200, n_train_obs=100, n_test_obs=100)
fits = fitting.batch_fit_models(setups)
metrics.batch_evaluate_models(fits)
print("Total time elapsed: %.0f seconds" % time.clock())
