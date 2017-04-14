import data_gen_utils as gen

setups = gen.batch_generate_setups(n_trans_factors=20, n_regulated_genes_per_trans_factor=10,
                                   n_tune_obs=200, n_train_obs=50, n_test_obs=50)
