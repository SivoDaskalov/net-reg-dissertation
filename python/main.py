import data_gen_utils as gen
import import_utils as imp
import model_fitting as fitting
import model_metrics as metrics
import plotting_utils as plut
import pandas as pd
import time

time.clock()

n_trans_factors = 20
n_regulated_genes_per_trans_factor = 10
p = n_trans_factors * (n_regulated_genes_per_trans_factor + 1)
# datasets = imp.batch_import_datasets()
# setups = gen.batch_generate_setups(n_regulated_genes_per_trans_factor=n_regulated_genes_per_trans_factor,
#                                    n_trans_factors=n_trans_factors, load_dump=True,
#                                    n_tune_obs=200, n_train_obs=100, n_test_obs=100)
#
# fits = fitting.batch_fit_models(setups, load_dump=True)
# results = metrics.batch_evaluate_models(fits)

results = metrics.load_results_from_csv(p)
# pd.set_option('display.width', 200)
# print(results)
plut.plot_results(results)

print("Total time elapsed: %.0f seconds" % time.clock())
