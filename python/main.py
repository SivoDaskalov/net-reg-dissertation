import data_gen_utils as gen
import model_fitting as fitting
import time
n_trans_factors = 200
n_regulated_genes_per_trans_factor = 10

setups = gen.batch_generate_setups(n_regulated_genes_per_trans_factor=n_regulated_genes_per_trans_factor,
                                   n_trans_factors=n_trans_factors, n_tune_obs=200, n_train_obs=50, n_test_obs=50)
fits = fitting.batch_fit_models(setups)
print("Total time elapsed: %f.0 seconds" % time.process_time())
