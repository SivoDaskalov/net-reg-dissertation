import data_gen_utils as gen
import model_fitting as fitting
import model_metrics as metrics
import pandas as pd
import time

n_trans_factors = 8
n_regulated_genes_per_trans_factor = 10

setups = gen.batch_generate_setups(n_regulated_genes_per_trans_factor=n_regulated_genes_per_trans_factor,
                                   n_trans_factors=n_trans_factors, load_dump=True,
                                   n_tune_obs=50, n_train_obs=25, n_test_obs=25)
fits = fitting.batch_fit_models(setups, load_dump=False)
results = metrics.batch_evaluate_models(fits)

pd.set_option('display.width', 200)
results = results.sort_values(['model', 'setup'])
print(results)
print("Total time elapsed: %.0f seconds" % time.clock())
