import data_gen_utils as gen
import model_fitting as fitting
import model_metrics as metrics
import pandas as pd
import time

n_trans_factors = 20
n_regulated_genes_per_trans_factor = 10

setups = gen.batch_generate_setups(n_regulated_genes_per_trans_factor=n_regulated_genes_per_trans_factor,
                                   n_trans_factors=n_trans_factors, load_dump=True,
                                   n_tune_obs=200, n_train_obs=100, n_test_obs=100)
# fits = fitting.batch_fit_models(setups, load_dump=True)
# results = metrics.batch_evaluate_models(fits)
#
# pd.set_option('display.width', 200)
# results = results.sort_values('model')
# print(results)
# print("Total time elapsed: %.0f seconds" % time.clock())

import matlab.engine
from models.grace import fit_grace, fit_py_grace

engine = matlab.engine.start_matlab("-nodesktop")
t_ = time.clock()
fit_grace(setups[0], engine)
m_ = time.clock()
print("Matlab took %.0f seconds" % (m_ - t_))
fit_py_grace(setups[0])
p_ = time.clock()
print("Python took %.0f seconds" % (p_ - m_))
